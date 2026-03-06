# CV Workshop — Image Classifier

**OpenCV + GoogLeNet (ImageNet)  ·  Beginner Computer Vision Workshop**

---

## Getting Started

First, you will need to fork this repo, ensuring your fork is public

Then, work on the code in this repo, try to get as much done as you can!

Finally, answer the questions listed at the bottom of the README to be entered into the raffle.

And most importantly, have fun!

## Setup

```bash
pip install opencv-python numpy
```

Then download the model files (one-time, ~50 MB):

```bash
python download_model.py
```

Sample images are already included in `images/`.

---

## Repo Structure

```
cv-workshop/
├── utils.py                    ← Part 1: implement this first
├── model.py                    ← Part 2: implement this second
├── main.py                     ← Part 3: wire everything together
├── download_model.py           ← run once to fetch model files
├── synset_words.txt            ← 1000 ImageNet labels
├── deploy.prototxt             ← model architecture  (after download)
├── bvlc_googlenet.caffemodel   ← model weights       (after download)
├── images/
│   ├── dog.jpg
│   ├── cat.jpg
│   ├── car.jpg
│   └── bird.jpg
```
---

## How to Work Through This

**Work in order: `utils.py` → `model.py` → `main.py`**

Each file has a self-test. Run it after you finish that file:

```bash
python utils.py    # must show all ✓ before moving to model.py
python model.py    # must show all ✓ before moving to main.py
python main.py     # runs the full pipeline
```

---

## Running the Classifier

```bash
# Default image
python main.py

# Specify image and expected label
python main.py --image images/cat.jpg --label cat
python main.py --image images/car.jpg --label "sports car"

# Classify every image in a folder
python main.py --batch images/
```

---

## The Pipeline

```
your_image.jpg
    ↓  load_image()
(H × W × 3)  BGR array
    ↓  preprocess()         grayscale → blur → Canny
(H × W)  binary edge map
    ↓  find_subject_contour()
largest qualifying contour
    ↓  crop_roi()
(h × w × 3)  color crop
    ↓  prepare_blob()
(1 × 3 × 224 × 224)  normalized tensor
    ↓  run_inference()
(1 × 1000)  confidence scores
    ↓  get_top_prediction()
"golden retriever"  94.3%
    ↓  draw_prediction()
annotated image on screen
```

---

## Some ImageNet Categories to Try

| Animals | Vehicles | Objects | Food |
|---------|----------|---------|------|
| golden retriever | sports car | laptop | pizza |
| tabby cat | school bus | backpack | banana |
| bald eagle | ambulance | rocking chair | ice cream |
| hammerhead shark | mountain bike | sunglasses | coffee mug |

Check `synset_words.txt` for the full list of 1000 valid labels.

---
## Questions - MUST BE DONE TO ENTER RAFFLE

1. In your own words, explain why we preprocess the image with grayscale, blur, and edge detection before passing it to the model. What would happen if we skipped one of those steps?

It reduces the image data to a manageable amount and allows for the data to be more parseable
by the model. If this didn't happen, the data would be a larger tensor than necessary to process the
proper classification.

2. When you ran your classifier on an image, what did it predict and how confident was it? Did the result surprise you — and if it got something wrong, why do you think that happened?

Due to time constraints and having to leave for a different meeting, 
I couldn't finish implementing the classifier. However, I imagine that it was decently confident. But,
with the decent amount of preprocessing done to reduce & feature extract the tensor data, I think
there could be a decent chance it got some stuff wrong. With less-reduced data, I image it might
be more accurate at the cost of compute.

3. We focused on the top prediction (the supposed classification) — but the model outputs 1000 scores simultaneously. What does it mean that the scores for other classes are non-zero? What are those numbers telling you?

Even with one best score, the other scores indicate confidence that it MIGHT be that thing.
The fact they are non-zero means they have potential to be those labels.

4. Where would you take this project next? Think about different models you could swap in, new kinds of images you'd want to classify, or features you'd add to make it more useful in the real world.

One could do try fine-tuning a model; instead of just calling a model after preprocessing some data,
I could work on trying to fine-tune a model for our particular dataset, such as something like
ResNet50, by adding top layer(s) to further train our model to be accurate for our data.

## Reference Docs

- OpenCV DNN:  https://docs.opencv.org/4.x/d6/d0f/group__dnn.html
- OpenCV all:  https://docs.opencv.org/4.x/
- NumPy:       https://numpy.org/doc/stable/reference/