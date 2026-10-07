# Integrated tiny-native group replays

The ordinary [converter](../../scripts/convert_coreml.py) built groups 8 and 9
with `--arithmetic ordered --softmax ordered --exp source --norm ordered --xsa
ordered --gelu lookup --queries dynamic-product --product paired --score-dot
tiny-native`. Its [group 8](integrated-group8/convert-8.json) and
[group 9](integrated-group9/convert-9.json) records identify the resulting
packages; logs are retained beside each record. The frozen score helper was
SHA-256 `c1fd3339ce3ac7b23cf0117dfb466b25c2c43c036a87616eb8656787fa5858ff`.

The [group-8 CPU_ONLY replay](integrated-group8/replay/results.json) used
`layer_31` and the other native boundary inputs from
`candidate4-tiny3-firstgroup/native-boundaries.npz`, with a new empty Core ML
state and the required prediction guard. Its `y` matched native `layer_35`
bitwise (0 differing, max absolute 0); `tap23_out` also matched the native
carried `layer_22` tensor bitwise. The generated
[input fixture](integrated-group8/native-fixture.npz) and package file hashes
are retained in the replay evidence.

The [group-9 CPU_ONLY replay](integrated-group9/replay/results.json) used the
saved first-failure inputs and restored every key/value cache from
`first-failure-state-before.npz` before prediction. Its `y` matched the saved
native `expected` tensor bitwise (0 differing, max absolute 0). That fixture
does not carry an independently named expected tap output, so this replay's
explicit pass claim is for `y`.

These are single saved-prefix group boundaries, not a complete end-to-end
candidate qualification. No shared math, runtime, builder, graph source, or
existing model package was edited for these replays.
