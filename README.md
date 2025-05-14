# modelsfromscratch
The intent of this repo is to build models from scratch to support posts on [https://mrcartoonology.github.io/](https://mrcartoonology.github.io/).

Currently it has the transformer for the post [Exploring the Transformer](https://mrcartoonology.github.io/jekyll/update/2025/05/14/exploring_the_transformer.html). 

The code was developed using mostly the original transformer paper and AI chats as reference. The main branch has a very interesting issue that
is part of the story on the post [Exploring the Transformer](https://mrcartoonology.github.io/jekyll/update/2025/05/14/exploring_the_transformer.html). 

Follow up work in the post is done in different branches. 

## Dependencies
I've been using `uv` for the projects.

## Notes
### Data
The transformer runs on all the python files from the pytorch repo, from commit `7a0781eaadd178a88fca6af826bb4990044ba6c8 ` dated  `Mon May 5 11:44:28 2025 -0700`.

## Run 
To run, edit the [config file](https://github.com/MrCartoonology/modelsfromscratch/blob/main/config/config.yaml)

and then do

```
uv run python src/modelsfromscratch/pipeline.py 
```

## Branches

* **attnvis** - this creates plots to visualize the attention mechanism. It hacks in returning the attention distributions from the model. It is run from
```
uv run python src/modelsfromscratch/attnvis.py
```
or import `modelsfromscratch.attnvis` into a `ipytnon` or `jupyter notebook` session and run interactively.

* **fix_rope** - this reruns `pipeline.py` with a fixed version of RoPE

* **attn_vis_fix_rope** - this reruns attnvis plots on the fixed rope. Some plots are done from [notebooks/transformer.ipynb](https://github.com/MrCartoonology/modelsfromscratch/blob/attnvis_fix_rope/notebooks/transformers.ipynb) from this branch.