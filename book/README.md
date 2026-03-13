Unfortunately, mdbook doesn't support links to rustdoc items yet.

In order to avoid hardcoding the full URL to doc items, and since we host the book and the docs at the same URL anyway, we assume the documentation is put under the src/ directory, such that we can use relative links in the document.

The CI will put it in there, so it works remotely.
To make it work locally, you can e.g. build the docs and soft link target/doc/ to book/src/doc (see the script `link_docs.sh`).