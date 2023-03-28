# rifs
The main entrypoint for interacting with the rifs project

## Useful commands
To build the image with docker:
```
docker build -t <tag> --ssh default .
```

To run the image:
```
docker run \
    -v data:/data \
    -v models:/models \
    -v output:/output \
    -v noise:/noise \
    <tag> [options] command
```
