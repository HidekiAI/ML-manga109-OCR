# Manga109 API for Rust

Rust version of [Manga109API](https://github.com/manga109/manga109api)

Still undecided on serde aspect (more specific, read-only deserialization).

There are (currently) no intentions to make this a public crate, mainly because I'm not too sure if it's useful.  It is used for the purpose of building data in preprocessing stage using Rust rather than Python.  And honestly, unlike Python (maybe I'm wrong, I purposely remain ignorant about Python), deserialization of XML to `struct` object is so trivial.
