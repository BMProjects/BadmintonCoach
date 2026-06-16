# third_party — vendored upstream projects (git submodules)

Adapters in `badminton_coach/perception/` wrap these. Each adapter's
`is_available()` checks for the corresponding directory here, so the pipeline
degrades gracefully when a submodule isn't initialised.

```bash
# Phase-1
git submodule add https://github.com/qaz812345/TrackNetV3 third_party/TrackNetV3
git submodule add https://github.com/jhwang7628/monotrack  third_party/monotrack

# Phase-2 (later)
git submodule add https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer third_party/BST

git submodule update --init --recursive
```

Do not edit upstream code in place — keep all glue in the adapters so upstream can
be updated by bumping the submodule.
