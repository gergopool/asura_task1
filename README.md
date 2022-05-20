# Asura ML Task

An example PyTorch solution for the introductory ML task.

## Environment

Make sure you have everything installed from *requirements.txt*.

## Download data

Install your csv and txt files under *resources/* folder. If your folder already exists somewhere else, don't worry, you can specify your custom path as the argparse suggest. For more info, run `./download --help`

In order to start downloading on multiple threads, run
```bash
./download --n-workers 4
```

## Train / Val / Test split

Split your data into three parts.

```bash
./split --folder resources/images
```

## Training

Example run:

```bash
./train --folder resources/images --model tiny
```

For more parameters, run `./train --help`