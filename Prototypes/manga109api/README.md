# Manga109 API for Rust

Rust version of [Manga109API](https://github.com/manga109/manga109api)

Still undecided on serde aspect (more specific, read-only deserialization).

There are (currently) no intentions to make this a public crate, mainly because I'm not too sure if it's useful.  It is used for the purpose of building data in preprocessing stage using Rust rather than Python.  And honestly, unlike Python (maybe I'm wrong, I purposely remain ignorant about Python), deserialization of XML to `struct` object is so trivial.

## Citation

```text
@article{multimedia_aizawa_2020,
    author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
    title={Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications},
    journal={IEEE MultiMedia},
    volume={27},
    number={2},
    pages={8--18},
    doi={10.1109/mmul.2020.2987895},
    year={2020}
}
```
