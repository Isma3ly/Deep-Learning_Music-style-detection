## Machine Learning : Music style detection

This project consists in developing a program in Python which will classify audio tracks according to the genre of music.

The amount of music streamed daily continues to skyrocket, especially on Internet platforms such as Soundcloud and Spotify, hence the need or the ability to instantly classify these songs into a playlist or library by genre has proven to be an important feature for any music streaming / purchase service.

The project is achieved; what is left now is only some updates to make the project look more simple and readable.

### Current baseline structure

At the moment, the repository is organized as follows:

- `attributes extraction/`: data download, metadata handling, and feature extraction scripts and utilities.
- `analysis notebooks/`: Jupyter notebooks for data loading, exploration, and feature visualization.
- `Models construction/`: model training, prediction scripts, and GUI-related code.
- `graphic interface build/`: an alternative PyQt-based graphical interface.
- `Rapport.pdf`: the original academic report describing the methodology and results.

Subsequent refactoring steps will build on this baseline layout while keeping the core functionality (music genre recognition from audio excerpts) intact.
