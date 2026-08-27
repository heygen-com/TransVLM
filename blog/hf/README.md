# Hugging Face blog post

Source of the TransVLM article written for the [`huggingface/blog`](https://github.com/huggingface/blog)
repository. Kept here so the submitted text and figures stay versioned alongside the code they
describe.

## Files

| File | Purpose |
|---|---|
| `transvlm.md` | The submission copy: YAML front matter, figures referenced as `images/...`. This is the file that goes to `huggingface/blog` as `transvlm.md`, with `images/` copied to `assets/transvlm/`. |
| `transvlm-remote-images.md` | The same body with no front matter and the figures pointed at their hosted URLs. Use it where the images cannot be uploaded alongside the text. |
| `_blog_yml_entry.yml` | The stanza to append to the `_blog.yml` at the root of `huggingface/blog`. Title, thumbnail and authors live in the article front matter and are deliberately not repeated there. |
| `images/` | The five figures as submitted. |

## Before submitting

The `authors` block in `transvlm.md` still carries the `YOUR_HF_USERNAME` and
`COAUTHOR_HF_USERNAME` placeholders. Fill in real Hugging Face usernames in the pull request
to `huggingface/blog`, not here.
