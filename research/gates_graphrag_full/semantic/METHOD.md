# Method

## Selection

The builder examines text in this order:

1. abstract;
2. introduction, preface or prologue;
3. conclusion, summary or outlook;
4. other text on physical PDF pages one through three.

It selects one relationship per full-text paper. An active author statement is
preferred over passive prose within the same section class. Supported cues
include statements such as "we show," "we define," "we construct," "we
study," and corresponding passive forms. Two papers without a qualifying cue
use the paper title as an explicit statement of subject.

The builder uses only relationships and entity types in the pilot controlled
vocabulary. It does not create a relationship from co-occurrence, vector
similarity or title similarity.

## Provenance

Every proposal records:

- paper identifier;
- extraction chunk identifier;
- physical PDF page;
- verbatim, whitespace-normalized evidence excerpt;
- extraction method;
- confidence;
- `pending` review status.

The validator confirms that every evidence excerpt occurs in its named chunk,
that the chunk belongs to the named paper and page, and that all 166 verified
local PDFs are covered once.

## Confidence

- Active author cues: 0.88 to 0.97
- Passive cues: 0.88 to 0.92
- Special result or evidence cues: 0.89 to 0.94
- Title-only subject fallback: 0.72

Confidence represents textual support for the proposed relationship. It does
not measure whether the scientific statement is correct.
