# Business case: BERT Sentiment Classifier

**Positive or negative, per review, from a small self-hosted model.**

## Who buys this

The buyer is a product or customer-insight lead at a streaming service, publisher, app studio or retailer that collects short English reviews. The users are analysts who need a sentiment label on every review, and developers who want it from an API they host themselves.

## The problem and what it costs

Review volume outgrows the people reading it, and sending every review to a paid third-party API adds cost and a data-sharing question. A small open model run in-house answers both. The model the app serves, `distilbert-base-uncased-finetuned-sst-2-english`, reports **91.1% accuracy** on the SST-2 validation set on its model card ([Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)).

I found no reliable public figure for what an unread review costs a business, so this case doesn't claim one.

## How the repo solves it

| Need | What in the repo does it |
|---|---|
| Score text over HTTP | `POST /predict` in the FastAPI service (`api/`) returns a label, confidence and positive probability per text |
| A page for non-developers | The React app (`web/`) behind nginx: paste reviews, read a verdict and a tally |
| Self-hosted weights | The Hugging Face model is pulled at `docker build`, so no review leaves your network at inference time |
| A path to a custom model | The original notebook and `trainer/` package fine-tune BERT and deploy it through a Vertex AI pipeline |

## Use-case scenarios

All of these organisations are fictional.

1. **A streaming service's content team** scores every new review of a title each morning and reads only the most negative ones.
2. **A mobile game studio** runs app-store reviews through `/predict` after each release to see whether a patch went down well.
3. **A small publisher** keeps reader comments on its own servers for privacy reasons and still gets a sentiment label on each.

## ROI: a worked model (illustrative)

These figures are **illustrative assumptions** unless a source is linked.

| Input | Value | Basis |
|---|---|---|
| Reviews a month | 20,000 | Assumption |
| Seconds for a person to read and label one | 20 | Assumption |
| Share of reviews a person still reads after scoring | 20% | Assumption (the most negative and the uncertain ones) |
| Analyst hourly cost | US$70 | BLS May 2024 median for data scientists, $54.13 an hour ([BLS OEWS](https://www.bls.gov/news.release/archives/ocwage_04022025.pdf)), multiplied by an assumed 1.3 for overhead |

Time without the model: 20,000 × 20 s ≈ 111 hours a month. With it, 20% of that, about 22 hours. Saving: about **89 hours a month, about US$75,000 a year**, against a small CPU or GPU host. The figure is only as good as the 20-second and 20% assumptions, so measure both on a pilot.

## Path to production

1. Evaluate the model on a labelled sample of your own reviews. SST-2 is movie-review sentences and may not match your domain.
2. If accuracy falls short, fine-tune with the notebook and `trainer/` path and register the model.
3. Add authentication, batching limits and request logging to the API.
4. Store scores with timestamps so trends can be reported.

## Risks and limits

- The model card itself reports biased predictions for some inputs, for example sentences that only differ by country. Test for this before acting on scores.
- Binary sentiment misses mixed reviews, sarcasm and the reason for a complaint.
- The repo's original fine-tuned weights were never published, so the local app uses a public substitute model.

**Case strength:** **Moderate.** Sentiment scoring is a real, common need, but it is also widely available off the shelf. The value here is a working, self-hosted starting point.

## Sources

- Hugging Face, distilbert-base-uncased-finetuned-sst-2-english model card: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- U.S. Bureau of Labor Statistics, OEWS May 2024: https://www.bls.gov/news.release/archives/ocwage_04022025.pdf
