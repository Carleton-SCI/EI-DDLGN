# TFHE-rs Subtree Maintenance

This directory contains a modified copy of the [TFHE-rs](https://github.com/zama-ai/tfhe-rs) library.
It is integrated into this repository using **Git Subtree**.

## Why Subtree?

Using a subtree allows us to:
1.  Keep the library code directly in our repository (no "repo inside repo" issues).
2.  Preserve our custom modifications.
3.  Easily pull updates from the official upstream repository.

## How to Update

To update the library to the latest version from Zama, run the helper script located in this directory:

```bash
./tfhe-rs/update_tfhe.sh
```

Or manually:

```bash
# 1. Add remote (if not exists)
git remote add tfhe-upstream https://github.com/zama-ai/tfhe-rs.git

# 2. Pull changes
git subtree pull --prefix=tfhe-rs tfhe-upstream main --squash
```

## Making Modifications

You can modify files in this directory just like any other file in the project. Git tracks them normally.

## Viewing Differences

To see how our version differs from the official upstream:

```bash
git fetch tfhe-upstream
git diff tfhe-upstream/main HEAD:tfhe-rs
```
