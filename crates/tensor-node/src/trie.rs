//! A Merkle-Patricia trie that keeps its nodes, so a block pays for what changed.
//!
//! The node rebuilt the whole trie every block. That is correct and it is why a large model
//! cannot go on chain: a block's cost is proportional to every account that has ever
//! existed, so an upload spends time proportional to blocks times accounts. Measured at
//! 0.53µs an account, a 61GB model's 2.6 million accounts would spend most of a day inside
//! the trie rather than in the upload.
//!
//! This keeps the trie between blocks and updates only the path from a changed leaf to the
//! root. A block that touches k accounts does O(k log n) work instead of O(n), which is what
//! every production client does and what makes the difference between a layer of a large
//! model and a whole one.
//!
//! The root it produces is Ethereum's, byte for byte — the tests check it against
//! `alloy-trie` on random sequences of inserts and removals, which is the point of having
//! kept a one-shot implementation to compare with.
//!
//! Keys are hashed account addresses, so every key is exactly 64 nibbles. That is worth
//! stating because it removes a whole class of case: no key can be a prefix of another, so
//! no branch node ever carries a value of its own.

use revm::primitives::{keccak256, B256};

/// Nibbles, one per byte, most significant first.
type Path = Vec<u8>;

fn nibbles(key: &B256) -> Path {
    let mut out = Vec::with_capacity(64);
    for b in key.iter() {
        out.push(b >> 4);
        out.push(b & 0x0f);
    }
    out
}

/// RLP for a byte string.
fn rlp_str(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len() + 9);
    if b.len() == 1 && b[0] < 0x80 {
        out.push(b[0]);
    } else if b.len() <= 55 {
        out.push(0x80 + b.len() as u8);
        out.extend_from_slice(b);
    } else {
        let len = b.len().to_be_bytes();
        let len = &len[len.iter().position(|&x| x != 0).expect("non-zero")..];
        out.push(0xb7 + len.len() as u8);
        out.extend_from_slice(len);
        out.extend_from_slice(b);
    }
    out
}

/// Wraps an already-encoded payload in an RLP list header.
fn rlp_list(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 9);
    if payload.len() <= 55 {
        out.push(0xc0 + payload.len() as u8);
    } else {
        let len = payload.len().to_be_bytes();
        let len = &len[len.iter().position(|&x| x != 0).expect("non-zero")..];
        out.push(0xf7 + len.len() as u8);
        out.extend_from_slice(len);
    }
    out.extend_from_slice(payload);
    out
}

/// The compact ("hex prefix") encoding of a path, with a flag for whether it ends a key.
fn compact(path: &[u8], leaf: bool) -> Vec<u8> {
    let flag = if leaf { 2u8 } else { 0 };
    let mut out = Vec::with_capacity(path.len() / 2 + 1);
    if path.len() % 2 == 1 {
        out.push(((flag + 1) << 4) | path[0]);
        for pair in path[1..].chunks(2) {
            out.push((pair[0] << 4) | pair[1]);
        }
    } else {
        out.push(flag << 4);
        for pair in path.chunks(2) {
            out.push((pair[0] << 4) | pair[1]);
        }
    }
    out
}

enum Node {
    Leaf {
        path: Path,
        value: Vec<u8>,
        /// The node's RLP, kept so an untouched subtrie is never re-encoded.
        enc: Option<Vec<u8>>,
    },
    Ext {
        path: Path,
        child: Box<Node>,
        enc: Option<Vec<u8>>,
    },
    Branch {
        children: Box<[Option<Node>; 16]>,
        enc: Option<Vec<u8>>,
    },
}

fn empty_children() -> Box<[Option<Node>; 16]> {
    Box::new([
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None,
    ])
}

impl Node {
    /// The node's RLP, computed once and cached until something beneath it changes.
    fn encode(&mut self) -> &[u8] {
        match self {
            Node::Leaf { path, value, enc } => {
                if enc.is_none() {
                    let mut p = rlp_str(&compact(path, true));
                    p.extend_from_slice(&rlp_str(value));
                    *enc = Some(rlp_list(&p));
                }
                enc.as_ref().expect("just set")
            }
            Node::Ext { path, child, enc } => {
                if enc.is_none() {
                    let c = child.reference();
                    let mut p = rlp_str(&compact(path, false));
                    p.extend_from_slice(&c);
                    *enc = Some(rlp_list(&p));
                }
                enc.as_ref().expect("just set")
            }
            Node::Branch { children, enc } => {
                if enc.is_none() {
                    let mut p = Vec::new();
                    for slot in children.iter_mut() {
                        match slot {
                            Some(n) => p.extend_from_slice(&n.reference()),
                            // An absent child is the empty string, not an empty list.
                            None => p.push(0x80),
                        }
                    }
                    // No key is a prefix of another here, so a branch never holds a value.
                    p.push(0x80);
                    *enc = Some(rlp_list(&p));
                }
                enc.as_ref().expect("just set")
            }
        }
    }

    /// How a parent refers to this node: its RLP inline when short, its hash otherwise.
    ///
    /// There is no separate invalidation step. Insert and remove rebuild every node on the
    /// path they touched and leave the rest alone, so a stale cache cannot survive a change
    /// beneath it — the node holding it was replaced.
    fn reference(&mut self) -> Vec<u8> {
        let enc = self.encode();
        if enc.len() < 32 {
            enc.to_vec()
        } else {
            rlp_str(keccak256(enc).as_slice())
        }
    }
}

/// Length of the longest common prefix of two paths.
fn shared(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Inserts `value` at `path`, returning the node that should stand in this position.
fn insert(node: Option<Node>, path: &[u8], value: Vec<u8>) -> Node {
    match node {
        None => Node::Leaf {
            path: path.to_vec(),
            value,
            enc: None,
        },

        Some(Node::Leaf {
            path: lp,
            value: lv,
            ..
        }) => {
            if lp == path {
                return Node::Leaf {
                    path: lp,
                    value,
                    enc: None,
                };
            }
            // The paths diverge. A branch at the first differing nibble, with whatever they
            // shared before it kept as an extension in front.
            let common = shared(&lp, path);
            let mut children = empty_children();
            children[lp[common] as usize] = Some(Node::Leaf {
                path: lp[common + 1..].to_vec(),
                value: lv,
                enc: None,
            });
            children[path[common] as usize] = Some(Node::Leaf {
                path: path[common + 1..].to_vec(),
                value,
                enc: None,
            });
            let branch = Node::Branch {
                children,
                enc: None,
            };
            if common == 0 {
                branch
            } else {
                Node::Ext {
                    path: lp[..common].to_vec(),
                    child: Box::new(branch),
                    enc: None,
                }
            }
        }

        Some(Node::Ext {
            path: ep,
            child,
            enc,
        }) => {
            let common = shared(&ep, path);
            if common == ep.len() {
                // Straight through: the extension still applies, only its child changes.
                let _ = enc;
                let next = insert(Some(*child), &path[common..], value);
                return Node::Ext {
                    path: ep,
                    child: Box::new(next),
                    enc: None,
                };
            }
            // The extension splits. Its remainder becomes a child of a new branch.
            let mut children = empty_children();
            let rest = ep[common + 1..].to_vec();
            children[ep[common] as usize] = Some(if rest.is_empty() {
                *child
            } else {
                Node::Ext {
                    path: rest,
                    child,
                    enc: None,
                }
            });
            children[path[common] as usize] = Some(Node::Leaf {
                path: path[common + 1..].to_vec(),
                value,
                enc: None,
            });
            let branch = Node::Branch {
                children,
                enc: None,
            };
            if common == 0 {
                branch
            } else {
                Node::Ext {
                    path: ep[..common].to_vec(),
                    child: Box::new(branch),
                    enc: None,
                }
            }
        }

        Some(Node::Branch { mut children, .. }) => {
            let i = path[0] as usize;
            let taken = children[i].take();
            children[i] = Some(insert(taken, &path[1..], value));
            Node::Branch {
                children,
                enc: None,
            }
        }
    }
}

/// Removes `path`, returning what should stand in this position.
fn remove(node: Option<Node>, path: &[u8]) -> Option<Node> {
    match node {
        None => None,
        Some(Node::Leaf {
            path: lp, value, ..
        }) => {
            if lp == path {
                None
            } else {
                Some(Node::Leaf {
                    path: lp,
                    value,
                    enc: None,
                })
            }
        }
        Some(Node::Ext {
            path: ep, child, ..
        }) => {
            if path.len() < ep.len() || path[..ep.len()] != ep[..] {
                return Some(Node::Ext {
                    path: ep,
                    child,
                    enc: None,
                });
            }
            match remove(Some(*child), &path[ep.len()..]) {
                None => None,
                // An extension onto an extension or a leaf collapses into one node, which
                // the trie must do or two states with the same contents would hash apart.
                Some(Node::Ext {
                    path: cp,
                    child: cc,
                    ..
                }) => Some(Node::Ext {
                    path: [ep, cp].concat(),
                    child: cc,
                    enc: None,
                }),
                Some(Node::Leaf {
                    path: cp, value, ..
                }) => Some(Node::Leaf {
                    path: [ep, cp].concat(),
                    value,
                    enc: None,
                }),
                Some(other) => Some(Node::Ext {
                    path: ep,
                    child: Box::new(other),
                    enc: None,
                }),
            }
        }
        Some(Node::Branch { mut children, .. }) => {
            let i = path[0] as usize;
            children[i] = remove(children[i].take(), &path[1..]);

            let live: Vec<usize> = (0..16).filter(|&j| children[j].is_some()).collect();
            match live.len() {
                0 => None,
                // A branch with one child left is no longer a branch: it becomes whatever
                // its child is, with the branch's nibble pushed onto the front of the path.
                1 => {
                    let j = live[0];
                    let only = children[j].take().expect("live child");
                    Some(match only {
                        Node::Leaf {
                            path: cp, value, ..
                        } => Node::Leaf {
                            path: [vec![j as u8], cp].concat(),
                            value,
                            enc: None,
                        },
                        Node::Ext {
                            path: cp, child, ..
                        } => Node::Ext {
                            path: [vec![j as u8], cp].concat(),
                            child,
                            enc: None,
                        },
                        branch => Node::Ext {
                            path: vec![j as u8],
                            child: Box::new(branch),
                            enc: None,
                        },
                    })
                }
                _ => Some(Node::Branch {
                    children,
                    enc: None,
                }),
            }
        }
    }
}

/// A trie of account leaves, keyed by hashed address.
#[derive(Default)]
pub struct Trie {
    root: Option<Node>,
}

impl Trie {
    pub fn new() -> Self {
        Self { root: None }
    }

    pub fn insert(&mut self, key: B256, value: Vec<u8>) {
        let path = nibbles(&key);
        let node = self.root.take();
        self.root = Some(insert(node, &path, value));
    }

    pub fn remove(&mut self, key: B256) {
        let path = nibbles(&key);
        let node = self.root.take();
        self.root = remove(node, &path);
    }

    /// The state root. Only nodes whose subtries changed are re-encoded.
    pub fn root(&mut self) -> B256 {
        match &mut self.root {
            // The empty trie is the hash of the RLP of the empty string, not zero.
            None => keccak256([0x80u8]),
            Some(n) => keccak256(n.encode()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_trie::root::state_root_unsorted;
    use alloy_trie::TrieAccount;
    use revm::primitives::U256;

    /// A leaf's value, encoded the way an account is.
    fn account(nonce: u64) -> (TrieAccount, Vec<u8>) {
        let a = TrieAccount {
            nonce,
            balance: U256::from(nonce) * U256::from(7u64),
            storage_root: keccak256([0x80u8]),
            code_hash: keccak256(nonce.to_be_bytes()),
        };
        let mut buf = Vec::new();
        alloy_rlp::Encodable::encode(&a, &mut buf);
        (a, buf)
    }

    fn key(i: u64) -> B256 {
        keccak256(i.to_be_bytes())
    }

    /// What `alloy-trie` says the root is, built from scratch.
    fn oracle(live: &[u64]) -> B256 {
        state_root_unsorted(live.iter().map(|&i| (key(i), account(i).0)))
    }

    #[test]
    fn empty_matches_the_ethereum_constant() {
        assert_eq!(
            format!("{:x}", Trie::new().root()),
            "56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"
        );
    }

    /// Inserting agrees with a trie built from scratch, at every size along the way.
    ///
    /// Checked after every insert rather than at the end, because a trie can be wrong in a
    /// way that only shows at a particular shape — a branch that should have collapsed, an
    /// extension that should have split — and the last state is unlikely to be that shape.
    #[test]
    fn insertion_agrees_with_a_trie_built_from_scratch() {
        let mut t = Trie::new();
        let mut live = Vec::new();
        for i in 0..400u64 {
            t.insert(key(i), account(i).1);
            live.push(i);
            assert_eq!(t.root(), oracle(&live), "after inserting {i}");
        }
    }

    /// Removal has to collapse branches and merge extensions, or two states with the same
    /// contents hash differently — which is a fork between a node that took a shortcut and
    /// one that did not.
    #[test]
    fn removal_agrees_with_a_trie_built_from_scratch() {
        let mut t = Trie::new();
        let mut live: Vec<u64> = (0..300).collect();
        for &i in &live {
            t.insert(key(i), account(i).1);
        }
        assert_eq!(t.root(), oracle(&live));

        // Removed in a scattered order, so branches empty out unevenly.
        let mut order: Vec<u64> = live.clone();
        order.sort_by_key(|&i| (i * 37) % 300);
        for i in order {
            t.remove(key(i));
            live.retain(|&j| j != i);
            assert_eq!(t.root(), oracle(&live), "after removing {i}");
        }
        assert_eq!(
            format!("{:x}", t.root()),
            "56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
            "an emptied trie must be the empty trie again"
        );
    }

    /// Updating a value in place, which is what most blocks actually do.
    #[test]
    fn updates_agree_with_a_trie_built_from_scratch() {
        let mut t = Trie::new();
        let live: Vec<u64> = (0..200).collect();
        for &i in &live {
            t.insert(key(i), account(i).1);
        }
        // Rewrite every account with a different nonce, and compare against an oracle built
        // from the same values.
        for &i in &live {
            t.insert(key(i), account(i + 1_000).1);
        }
        let want = state_root_unsorted(live.iter().map(|&i| (key(i), account(i + 1_000).0)));
        assert_eq!(t.root(), want);
    }

    /// Inserting and removing in a shuffled sequence, which is what a chain does.
    #[test]
    fn interleaved_inserts_and_removals_agree() {
        let mut t = Trie::new();
        let mut live: Vec<u64> = Vec::new();
        let mut x = 12345u64;
        for step in 0..600 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let i = x % 150;
            if live.contains(&i) && step % 3 == 0 {
                t.remove(key(i));
                live.retain(|&j| j != i);
            } else if !live.contains(&i) {
                t.insert(key(i), account(i).1);
                live.push(i);
            }
            assert_eq!(t.root(), oracle(&live), "at step {step}");
        }
    }

    /// Removing something that was never there must not change the root.
    #[test]
    fn removing_an_absent_key_is_a_no_op() {
        let mut t = Trie::new();
        let live: Vec<u64> = (0..50).collect();
        for &i in &live {
            t.insert(key(i), account(i).1);
        }
        let before = t.root();
        t.remove(key(9_999));
        assert_eq!(t.root(), before);
    }
}
