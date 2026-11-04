import re
import sys
from pathlib import Path
import json
import random
import time

import pandas as pd

# Make the project root available for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm


INPUT_FILE = ROOT / "data" / "raw" / "tickets.csv"
MAX_INPUT_TOKENS = 400

# Required privacy patterns
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"


def redact_text(text):
    """Redact email, phone number and order ID in the required order."""

    text, email_count = re.subn(EMAIL_PATTERN, "[EMAIL]", text)
    text, phone_count = re.subn(PHONE_PATTERN, "[PHONE]", text)
    text, order_count = re.subn(ORDER_PATTERN, "[ORDER_ID]", text)

    return text, email_count, phone_count, order_count

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

BASE_DELAY = 0.05
MAX_RETRIES = 3

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category:
billing, delivery, technical, fraud.

Return JSON only in this format:
{"category": "...", "reason": "..."}
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category:
billing, delivery, technical, fraud.

Return JSON only in this format:
{"category": "...", "reason": "..."}

EXAMPLES:
Example 1:
Ticket: I was charged twice for my order.
Answer: {"category": "billing", "reason": "Duplicate charge"}

Example 2:
Ticket: Someone accessed my account without permission.
Answer: {"category": "fraud", "reason": "Unauthorized account access"}
"""


def call_with_retry(ticket_id, messages):
    """Call the mock model with exponential backoff and jitter."""

    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise

            wait_time = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Retry: ticket={ticket_id}, "
                f"retry_number={retry_number}, "
                f"wait={wait_time:.5f}s"
            )

            time.sleep(wait_time)

def validate_response(response_text):
    """Validate JSON output or extract exactly one category keyword."""

    try:
        result = json.loads(response_text)

        if (
            isinstance(result, dict)
            and result.get("category") in ALLOWED_CATEGORIES
        ):
            return result["category"], "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", response_text.lower())

        matches = ALLOWED_CATEGORIES.intersection(words)

        if len(matches) == 1:
            return matches.pop(), "classified_keyword"

        return "", "flagged_invalid_output"

def run_classification(tickets, system_prompt, token_budget=None):
    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        text = ticket["redacted_text"]

        if not text.strip():
            status = "skipped_empty"
            category = ""
            tokens_used = 0

        elif mock_llm.est_tokens(text) > 400:
            status = "flagged_long"
            category = ""
            tokens_used = 0

        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
            category = ""
            tokens_used = 0

        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]

            response = call_with_retry(ticket_id, messages)
            response_text = response.choices[0].message.content

            category, status = validate_response(response_text)
            tokens_used = response.usage.total_tokens
            total_tokens += tokens_used

        results.append({
            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": status,
            "category": category,
            "tokens_used": tokens_used
        })

    return pd.DataFrame(results), total_tokens    

TOKEN_BUDGET = 100000

def main():
    tickets = pd.read_csv(
        ROOT / "data" / "raw" / "tickets.csv",
        keep_default_na=False
    )

    tickets["redacted_text"] = tickets["text"].apply(
    lambda text: redact_text(text)[0]
)

    # Verify that all personal information has been removed.
    combined_text = " ".join(tickets["redacted_text"])
    assert not re.search(EMAIL_PATTERN, combined_text)
    assert not re.search(PHONE_PATTERN, combined_text)
    assert not re.search(ORDER_PATTERN, combined_text)

    print("Running zero-shot classification...")
    zero_shot_results, zero_shot_tokens = run_classification(
        tickets, ZERO_SHOT_PROMPT
    )

    print("\nRunning few-shot classification...")
    few_shot_results, few_shot_tokens = run_classification(
        tickets, FEW_SHOT_PROMPT
    )

    print("\nZero-shot status counts:")
    print(zero_shot_results["status"].value_counts())

    print("\nFew-shot status counts:")
    print(few_shot_results["status"].value_counts())

    print("\nZero-shot total tokens:", zero_shot_tokens)
    print("Few-shot total tokens:", few_shot_tokens)

        # Evaluate only tickets that received a valid category.
    zero_valid = zero_shot_results[
        zero_shot_results["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    few_valid = few_shot_results[
        few_shot_results["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    zero_accuracy = (
        zero_valid["category"] == zero_valid["true_category"]
    ).mean()

    few_accuracy = (
        few_valid["category"] == few_valid["true_category"]
    ).mean()

    print("\nZero-shot classified tickets:", len(zero_valid))
    print(f"Zero-shot accuracy: {zero_accuracy:.1%}")

    print("\nFew-shot classified tickets:", len(few_valid))
    print(f"Few-shot accuracy: {few_accuracy:.1%}")

    print("\nZero-shot confusion matrix:")
    print(pd.crosstab(
        zero_valid["true_category"],
        zero_valid["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False
    ))
    print("\nRunning zero-shot with 6000-token budget...")

    budget_results, budget_tokens = run_classification(
        tickets,
        ZERO_SHOT_PROMPT,
        token_budget=6000
    )

    budget_results.to_csv(
        ROOT / "outputs" / "triage_budget_demo.csv",
        index=False
    )

    print("\nBudget demonstration status counts:")
    print(budget_results["status"].value_counts())

    print("\nTotal tokens used:", budget_tokens)
    print("Budget output saved to outputs/triage_budget_demo.csv")

    print("\nRunning final few-shot triage...")

    final_results, final_tokens = run_classification(
        tickets,
        FEW_SHOT_PROMPT,
        token_budget=TOKEN_BUDGET
    )

    # Load customer segments.
    segments = pd.read_csv(
        ROOT / "outputs" / "customer_segments.csv"
    )

    # Attach customer segment information.
    final_results = final_results.merge(
        segments[["customer_id", "segment_name", "is_premium"]],
        on="customer_id",
        how="left"
    )

    final_results["segment_name"] = (
        final_results["segment_name"].fillna("")
    )

    # Apply ticket routing rules.
    def assign_route(row):
        if row["status"] in [
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted"
        ]:
            return "NORMAL", "human_review"

        if row["category"] == "fraud":
            return "HIGH", "security_team"

        if row["is_premium"] == True:
            return "HIGH", "premium_desk"

        return "NORMAL", "support_queue"

    final_results[["priority", "route"]] = final_results.apply(
        lambda row: pd.Series(assign_route(row)),
        axis=1
    )

    # Save only the required columns.
    final_results = final_results[
        [
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used"
        ]
    ]

    final_results.to_csv(
        ROOT / "outputs" / "triage_results.csv",
        index=False
    )

    print("\nFinal triage status counts:")
    print(final_results["status"].value_counts())

    print("\nFinal routing counts:")
    print(final_results["route"].value_counts())

    print("\nFinal tokens used:", final_tokens)
    print("Saved outputs/triage_results.csv")

if __name__ == "__main__":
    main()