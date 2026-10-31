import io
import json

import pytest
from fastapi.testclient import TestClient

from clef_tune.ui.server import create_app
from clef_tune.ui.settings import Settings
from tests.helpers import ticket_rows, write_csv
from tests.ui.fakes import fake_services

LOCAL = Settings(space=False, max_upload_mb=1, local_training=False)
SCRIPT = {"X-Requested-With": "clef-tune"}


@pytest.fixture
def services_and_calls(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_machine")  # the machine's login, used outside a Space
    return fake_services()


@pytest.fixture
def client(services_and_calls):
    app = create_app(LOCAL, services_and_calls[0], secret="test")
    with TestClient(app) as c:
        yield c


def sample(client, name="support_tickets"):
    response = client.post("/data/sample", data={"name": name}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/data"


def test_the_welcome_page_greets_a_newcomer(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("clef_tune.ui.settings.get_token", lambda: None)
    app = create_app(LOCAL, fake_services()[0], secret="test")
    with TestClient(app) as c:
        page = c.get("/").text
    assert "Bespoke intelligence, <em>owned by you</em>." in page
    assert "Fine-tuned decisions for everyone" in page and "claim triage" in page
    assert "YOUR DECISION" in page and "Your data stays yours" in page
    assert "how much better it got" not in page and "how accurate it is" in page
    # "decision model" explains itself, and the footer points to Cloudflare's own page.
    assert 'aria-describedby="decision-model-tip"' in page and "behind chatbots" in page
    tip = page.split('id="decision-model-tip">')[1].split("</span>")[0]
    assert "Clef" not in tip.split("<a ")[0] and ">Learn more</a>" in tip
    guide = 'href="https://www.datacamp.com/tutorial/what-are-ai-decision-models"'
    assert f'{guide} target="_blank"' in tip  # a general explainer, not a vendor page
    assert (
        page.count('href="https://blog.cloudflare.com/clef-decision-models/" target="_blank"') == 1
    )  # the footer
    assert 'action="/signout"' not in page  # on this computer, its own login is used


def test_a_returning_person_sees_their_latest_models(client):
    page = client.get("/").text
    assert "Hello again" in page and "run-a" in page and "See all 4 models" in page


def test_later_steps_wait_for_data(client):
    for path in ("/questions", "/settings", "/train"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 303 and response.headers["location"] == "/data"


def test_the_data_step_shows_the_file_columns_and_checks(client):
    assert "Drop a spreadsheet here" in client.get("/data").text
    sample(client)
    page = client.get("/data").text
    assert "support_tickets.csv" in page and 'value="ticket"' in page
    assert "Guessed from the file name" in page
    assert 'class="tag chev r-question"' in page and "Expected output" in page
    assert "All 200 rows read, with 7 columns" in page and "3 expected outputs" in page
    assert 'href="/questions"' in page


def test_changing_a_role_swaps_only_the_data_body_and_steps(client):
    sample(client)
    response = client.post(
        "/data/role", data={"column": "customer_tier", "role": "skip"}, headers=SCRIPT
    )
    body = response.text
    assert body.startswith('<template data-swap="#data-body">')
    assert '<template data-swap="#steps-slot">' in body
    assert "2 columns read" in body


def test_the_noun_updates_the_guess_marker(client):
    sample(client)
    assert client.post("/data/noun", data={"noun": "case"}, headers=SCRIPT).text == (
        '<template data-swap="#noun-mark"></template>'
    )
    assert "Guessed" in client.post("/data/noun", data={"noun": "ticket"}, headers=SCRIPT).text


def test_uploads_work_and_oversized_ones_explain_why(client, tmp_path):
    csv_path = write_csv(tmp_path / "claims.csv", ticket_rows(30))
    with open(csv_path, "rb") as handle:
        client.post("/data/upload", files={"file": ("claims.csv", handle, "text/csv")})
    page = client.get("/data").text
    assert "claims.csv" in page and 'value="claim"' in page
    big = io.BytesIO(b"x" * (2 * 2**20))
    shown = client.post("/data/upload", files={"file": ("huge.csv", big, "text/csv")})
    assert "over the 1 MB limit" in shown.text  # shown once, on the page it redirects to
    assert "over the 1 MB limit" not in client.get("/data").text


def test_the_data_can_be_removed_to_choose_another_sample(client):
    sample(client)
    assert 'action="/data/remove"' in client.get("/data").text
    response = client.post("/data/remove", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/data"
    page = client.get("/data").text
    assert "Drop a spreadsheet here" in page and "support_tickets.csv" not in page
    assert client.get("/questions", follow_redirects=False).headers["location"] == "/data"
    sample(client, "claims_triage")
    assert "claims_triage.csv" in client.get("/data").text


def test_column_names_are_escaped(client, tmp_path):
    rows = [{"<script>x</script>": "a", "label": "y" if i % 2 else "n"} for i in range(30)]
    with open(write_csv(tmp_path / "evil.csv", rows), "rb") as handle:
        client.post("/data/upload", files={"file": ("evil.csv", handle, "text/csv")})
    page = client.get("/data").text
    assert "&lt;script&gt;x&lt;/script&gt;" in page and "<script>x</script>" not in page


def test_the_question_step_edits_one_question_at_a_time(client):
    sample(client)
    page = client.get("/questions?q=urgency").text
    assert "What should the model learn to predict?" in page
    assert "How quickly does this ticket need a response?" in page
    assert "Lowest first" in page and 'aria-describedby="ask-tip"' in page
    swapped = client.post("/questions/urgency/type", data={"kind": "choice"}, headers=SCRIPT)
    assert swapped.text.startswith('<template data-swap="#editor">')
    flags = client.post("/questions/urgency/ask", data={"ask": "How soon?"}, headers=SCRIPT)
    assert flags.text.startswith('<template data-swap="#flags-2">')
    assert "How soon?" in client.get("/questions?q=urgency").text


def test_switching_questions_swaps_only_the_list_and_the_editor(client):
    sample(client)
    assert "data-nav" in client.get("/questions").text
    swapped = client.get("/questions?q=urgency", headers=SCRIPT).text
    assert swapped.startswith('<template data-swap="#qlist">')
    assert '<template data-swap="#qswitch">' in swapped
    assert '<template data-swap="#editor" data-animate>' in swapped
    assert "<h1" not in swapped and "How quickly does this ticket need a response?" in swapped


def test_the_prompt_preview_is_escaped_and_marked(client):
    sample(client)
    prompt = client.get("/questions/prompt").text
    assert 'class="k-state"' in prompt and "&lt;|im_start|&gt;" in prompt


def test_settings_show_the_estimate_and_update_in_place(client):
    sample(client)
    page = client.get("/settings").text
    assert "How should it be trained?" in page and "never cost more than" in page
    assert "Saved to your Hugging Face account as <b>me/support-tickets</b>" in page
    swapped = client.post("/settings/preset", data={"preset": "quick"}, headers=SCRIPT)
    assert swapped.text.startswith('<template data-swap="#settings-body">')
    assert "Quick test" in swapped.text
    bad = client.post(
        "/settings/field", data={"path": "train.max_length", "value": "100"}, headers=SCRIPT
    )
    row = bad.text.split('value="train.max_length"')[1].split("</form>")[0]
    assert 'class="err"' in row and "Use a whole number: 256 to 16,384." in row
    typo = client.post(
        "/settings/field", data={"path": "train.lora_lr", "value": "fast"}, headers=SCRIPT
    )
    assert "Use a number: more than 0." in typo.text


def test_advanced_settings_explain_themselves(client):
    sample(client)
    page = client.get("/settings").text
    assert "Usually 8–64" in page and 'aria-label="About Adapter size"' in page
    assert "Allowed: 256 to 16,384." in page
    assert "data.eval_fraction" not in page


def test_the_test_share_is_edited_in_the_holdout_sentence(client):
    sample(client)
    page = client.get("/settings").text
    assert 'name="percent" value="15"' in page and "31 in all" in page
    swapped = client.post("/settings/holdout", data={"percent": "25"}, headers=SCRIPT).text
    assert swapped.startswith('<template data-swap="#settings-body">')
    assert 'name="percent" value="25"' in swapped and "31 in all" not in swapped
    refused = client.post("/settings/holdout", data={"percent": "70"}, headers=SCRIPT).text
    assert "Choose a share from 5% to 50%" in refused and 'name="percent" value="70"' in refused


def test_training_is_reviewed_then_started(client, services_and_calls):
    _, calls = services_and_calls
    sample(client)
    review = client.get("/train/review").text
    assert "About 21–45 minutes" in review and "never costs more than <b>$0.80</b>" in review
    assert "200 tickets · 3 questions" in review and "Signed in as <b>me</b>" in review
    started = client.post("/train/start", follow_redirects=False)
    assert started.headers["location"] == "/runs/job-new"
    request = next(c for c in calls if c[0] == "plan_launch")[1][1]
    assert request.adapter_repo == "me/support-tickets" and request.keep_data is False


def test_a_running_job_polls_its_status(client):
    page = client.get("/runs/job-a").text
    assert "Training run-a" in page and 'data-poll="/runs/job-a/status"' in page
    assert "Learning from your examples" in page and "You can close this page" in page
    status = client.get("/runs/job-a/status")
    assert status.text.startswith('<template data-swap="#run-status">')
    assert "X-Refresh" not in status.headers


def test_a_finished_job_shows_its_results(client):
    page = client.get("/runs/job-b").text
    assert "run-b is ready" in page
    assert "It now matches every expected output on <b>24 of 30</b>" in page
    assert "Which team should handle this ticket?" in page and 'href="/predict/job-b"' in page
    assert 'class="dg after"' in page


def test_a_failed_job_offers_to_try_again(client):
    page = client.get("/runs/job-c").text
    assert "run-c didn&#39;t finish" in page and 'data-load="/runs/job-c/resume"' in page
    quote = client.get("/runs/job-c/resume").text
    assert "Try again from where it stopped" in quote and "at most $1.60" in quote
    resumed = client.post("/runs/job-c/resume", follow_redirects=False)
    assert resumed.headers["location"] == "/runs/job-resumed"


def test_finished_predictions_can_be_downloaded(client):
    page = client.get("/runs/job-p").text
    assert "3 tickets predicted" in page and "1 ticket is marked for review" in page
    assert page.count('class="btn primary"') == 1 and ">Download ↓</button>" in page
    assert "Download as Excel" not in page and 'data-menu="#download-menu"' in page
    for fmt in ("xlsx", "csv", "tsv", "jsonl", "json"):
        assert f'href="/runs/job-p/download.{fmt}"' in page
    assert "What it answered" in page
    csv = client.get("/runs/job-p/download.csv")
    assert csv.status_code == 200 and "T2041" in csv.text
    xlsx = client.get("/runs/job-p/download.xlsx")
    assert xlsx.status_code == 200 and xlsx.content[:2] == b"PK"
    jsonl = client.get("/runs/job-p/download.jsonl")
    assert jsonl.headers["content-type"].startswith("application/x-ndjson")
    assert 'filename="run-b-predict-predictions.jsonl"' in jsonl.headers["content-disposition"]
    assert [json.loads(line)["ticket_id"] for line in jsonl.text.splitlines()][0] == "T2041"
    assert client.get("/runs/job-p/download.json").json()[0]["ticket_id"] == "T2041"
    assert "\t" in client.get("/runs/job-p/download.tsv").text
    unknown = client.get("/runs/job-p/download.exe", follow_redirects=False)
    assert unknown.status_code == 303


def test_your_models_lists_everything_with_a_lazy_line(client):
    page = client.get("/models").text
    assert "Your models" in page and "Predictions ready" in page
    assert 'data-load="/models/job-b/line"' in page
    line = client.get("/models/job-b/line").text
    assert "It now matches every expected output on 24 of 30 tickets" in line


def test_a_new_spreadsheet_is_checked_priced_and_predicted(client, tmp_path):
    assert "Predict with run-b" in client.get("/predict/job-b").text
    rows = [{"subject": "s", "body": "b", "customer_tier": "pro"} for _ in range(5)]
    with open(write_csv(tmp_path / "new.csv", rows), "rb") as handle:
        client.post("/predict/job-b/upload", files={"file": ("new.csv", handle, "text/csv")})
    page = client.get("/predict/job-b").text
    assert "Found subject, body and customer_tier" in page and "numbered from 1" in page
    assert client.post("/predict/job-b/threshold", data={"percent": "80"}).status_code == 204
    estimate = client.get("/predict/job-b/estimate").text
    assert "never costs more than <b>$0.90</b>" in estimate and "Start predicting" in estimate
    started = client.post("/predict/job-b/start", follow_redirects=False)
    assert started.headers["location"] == "/runs/job-pnew"


def test_a_file_missing_a_column_can_be_mapped(client, tmp_path):
    client.get("/predict/job-b")
    rows = [{"title": "s", "body": "b", "customer_tier": "pro"} for _ in range(3)]
    with open(write_csv(tmp_path / "renamed.csv", rows), "rb") as handle:
        client.post("/predict/job-b/upload", files={"file": ("renamed.csv", handle, "text/csv")})
    assert "Couldn&#39;t find subject" in client.get("/predict/job-b").text
    fixed = client.post(
        "/predict/job-b/rename", data={"from": "title", "to": "subject"}, headers=SCRIPT
    ).text
    assert "Found subject, body and customer_tier" in fixed


SPACE = Settings(space=True, max_upload_mb=200, local_training=False)


@pytest.fixture
def mocked_oauth(monkeypatch):
    """Outside a real Space, huggingface_hub mocks sign-in from the machine's own login."""
    import huggingface_hub._oauth as oauth

    monkeypatch.setattr(oauth, "get_token", lambda: "hf_local_dev")
    monkeypatch.setattr(
        oauth,
        "whoami",
        lambda *a, **k: {
            "type": "user",
            "name": "dev",
            "fullname": "Dev",
            "email": None,
            "avatarUrl": "",
            "orgs": [],
        },
    )


def test_on_a_space_visitors_sign_in_and_the_server_token_is_never_used(monkeypatch, mocked_oauth):
    monkeypatch.setenv("HF_TOKEN", "hf_server_owner")
    services, calls = fake_services()
    app = create_app(SPACE, services)
    with TestClient(app) as c:
        assert "Sign in with Hugging Face" in c.get("/").text
        sample(c)
        review = c.get("/train/review").text
        assert "Sign in to train" in review and "disabled" in review
        assert "Sign in with Hugging Face to see the models" in c.get("/models").text
    assert not [call for call in calls if call[0] in ("plan_launch", "list_jobs", "whoami")]


def test_signing_out_on_a_space_forgets_the_visitor(monkeypatch, mocked_oauth):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with TestClient(create_app(SPACE, fake_services()[0]), base_url="https://testserver") as c:
        newcomer = c.get("/").text
        assert "When you're ready to train, you can sign in with Hugging Face." in newcomer
        c.get("/oauth/huggingface/login?_target_url=/")
        sample(c)
        signed_in = c.get("/").text
        assert 'action="/signout"' in signed_in and ">Sign in with Hugging Face<" not in signed_in
        out = c.post("/signout", follow_redirects=False)
        assert out.headers["location"] == "/oauth/huggingface/logout?_target_url=/"
        assert ">Sign in with Hugging Face<" in c.get(out.headers["location"]).text
        assert "Drop a spreadsheet here" in c.get("/data").text  # the setup went with them


def test_a_pasted_token_is_the_fallback_when_sign_in_lacks_a_permission(monkeypatch, mocked_oauth):
    from clef_tune.launch.hf_jobs import LaunchError

    monkeypatch.setenv("HF_TOKEN", "hf_server_owner")
    seen = []

    def plan(config_path, request, token=None, **kwargs):
        seen.append(token)
        raise LaunchError("could not upload the bundle: 403; the token may lack a permission")

    services, _ = fake_services(plan_launch=plan)
    # A Space's session cookie is HTTPS-only (SameSite=None; Secure), as on huggingface.co.
    with TestClient(create_app(SPACE, services), base_url="https://testserver") as c:
        sample(c)
        c.post("/train/token", data={"token": " hf_pasted "})
        review = c.get("/train/review").text
    assert seen == ["hf_pasted"] and "lack a permission" in review
    assert 'name="token"' in review and 'type="password"' in review
