# Historical adapter prototypes

These files were moved out of `src/mesher/mesh2d/adapter` when the active adapter
was organized into planning, construction, and repair packages.

- `old/` preserves the previous experimental implementation verbatim. It
  includes incomplete functions and imports from an obsolete environment.
- `comp/` preserves the empty wrapper placeholders, including the original
  `wrap_outer,py` filename.

This archive is reference material, not an importable or supported package.
It is outside the setuptools `src/` package-discovery root and is not shipped
in the wheel. Do not add runtime imports from this directory. Active code lives
in `src/mesher/mesh2d/adapter`; see
[the package architecture](../../docs/architecture.md) for its responsibilities
and public import paths.
