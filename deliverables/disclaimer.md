# UI disclaimer and refusal copy

Every string on this page is the exact text the app renders. They are defined once
in `src/config.py` and imported by the UI, the eval harness and the tests, so this
file and the running app cannot drift apart.

## Welcome line

> Ask factual questions about 5 HDFC mutual fund schemes.

## Disclaimer (always visible)

> Facts-only. No investment advice.

Rendered as an `st.info(...)` note at the top of the page and as the last line of
every assistant turn.

## Freshness stamp

Every sourced answer ends with:

> Last updated from sources: &lt;date&gt;

The prefix is fixed (`Last updated from sources: `); the date is `DD Mon YYYY`
taken from the retrieved chunk's `fetched_at` metadata, never from the model.

## Example questions

1. What is the expense ratio of HDFC Large Cap Direct Growth?
2. What is the lock-in period in HDFC ELSS Tax Saver Fund?
3. How do I download my capital gains statement?

## Refusal templates

A refusal renders in an info block with its one educational link and **no**
citation badge and **no** freshness stamp. Each template is at most three
sentences and contains no digits.

### Advice

> I can share published facts about these schemes, but I can't give investment
> advice or suggest whether to buy or sell. For a view on whether a scheme suits
> you, please read its Key Information Memorandum or consult a SEBI-registered
> investment adviser.

### Performance

> I don't calculate, estimate, rank, or compare returns, because past performance
> doesn't predict future returns. You can check the scheme's official factsheet
> for the published figures.

### Personal data (PII)

> I can't process personal details, so please don't share PAN, Aadhaar, account
> numbers, one-time passwords, email addresses, or phone numbers here. Once you
> remove those, I can answer factual questions about the schemes.

### Out of scope (another AMC)

> I only have facts for the schemes in my sources, and that fund isn't one of
> them. For any other fund, please check the fund's official page or its AMC's
> disclosures.

### Low confidence (not in the sources)

> I couldn't find that in the scheme pages I was built from, so I don't want to
> guess. Here's the official scheme page so you can check the detail yourself.

### Live NAV

> I don't have live NAV or price data, because my sources are a static snapshot
> of the scheme pages. You can check the current NAV on the official scheme page.

### Unknown intent (scheme found, fact not)

> I could find the scheme page, but not that specific detail in it. I'd rather
> not answer from memory, so please check the official scheme page or the
> scheme's Key Information Memorandum.
