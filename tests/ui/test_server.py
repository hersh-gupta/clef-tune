import asyncio
import dataclasses
import io
import json
from datetime import datetime, timedelta
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from clef_tune.ui.server import create_app
from clef_tune.ui.settings import Settings
from tests.helpers import ticket_rows, write_csv
from tests.ui import fakes
from tests.ui.fakes import fake_services

LOCAL = Settings(space=False, max_upload_mb=1, local_training=False)
SCRIPT = {"X-Requested-With": "clef-tune"}
HERE = "http://localhost"  # on this computer the app answers only to its own names


@pytest.fixture(autouse=True)
def machine_login(monkeypatch):
    """This computer's Hugging Face login, used outside a Space: a fake one, so no test reads
    the real one (CI has none). logged_out takes it away."""
    monkeypatch.setenv("HF_TOKEN", "hf_machine")


@pytest.fixture
def services_and_calls():
    return fake_services()


@pytest.fixture
def client(services_and_calls):
    app = create_app(LOCAL, services_and_calls[0], secret="test")
    with TestClient(app, base_url=HERE) as c:
        yield c


@pytest.fixture
def logged_out(monkeypatch):
    """This computer has no Hugging Face login."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("clef_tune.ui.settings.get_token", lambda: None)


def local_client(services, **kwargs):
    return TestClient(create_app(LOCAL, services, secret="test", **kwargs), base_url=HERE)


def sample(client, name="support_tickets"):
    response = client.post("/data/sample", data={"name": name}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/data"


def test_the_welcome_page_greets_a_newcomer(logged_out):
    with local_client(fake_services()[0]) as c:
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
    swapped = client.post(
        "/questions/type", data={"qid": "urgency", "kind": "choice"}, headers=SCRIPT
    )
    assert swapped.text.startswith('<template data-swap="#editor">')
    flags = client.post(
        "/questions/ask", data={"qid": "urgency", "ask": "How soon?"}, headers=SCRIPT
    )
    assert flags.text.startswith('<template data-swap="#flags-2">')
    assert "How soon?" in client.get("/questions?q=urgency").text


def test_a_question_named_with_a_slash_can_be_edited(client, tmp_path):
    answers = ("approved", "declined", "pending")
    rows = [{"note": f"case {i}", "Approved Y/N": answers[i % 3]} for i in range(40)]
    with open(write_csv(tmp_path / "approvals.csv", rows), "rb") as handle:
        client.post("/data/upload", files={"file": ("approvals.csv", handle, "text/csv")})
    client.post("/data/role", data={"column": "Approved Y/N", "role": "question"}, headers=SCRIPT)
    qid = "Approved Y/N"
    page = client.get(f"/questions?q={quote(qid)}").text
    assert 'name="qid" value="Approved Y/N"' in page and "/questions/Approved" not in page
    asked = client.post("/questions/ask", data={"qid": qid, "ask": "Was it approved?"})
    assert asked.status_code == 200
    typed = client.post("/questions/type", data={"qid": qid, "kind": "score"}, headers=SCRIPT)
    assert typed.status_code == 200 and "Lowest first" in typed.text
    described = {"qid": qid, "option": "approved", "description": "signed off"}
    assert client.post("/questions/describe", data=described).status_code == 200
    moved = {"qid": qid, "option": "pending", "offset": "-1"}
    assert client.post("/questions/move", data=moved).status_code == 200
    page = client.get(f"/questions?q={quote(qid)}").text
    assert "Was it approved?" in page and 'value="signed off"' in page
    editor = page.split('id="editor"')[1]
    assert editor.index('value="pending"') < editor.index('value="declined"')
    assert 'class="on" aria-pressed="true">Scale' in editor


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


# ---------------------------------------------------------------------------- request safety


def test_posts_from_another_site_are_refused(client):
    sample(client)
    for headers in (
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Origin": "https://evil.example"},
        {"Origin": "http://localhost:9999"},
        {"Origin": "null"},
    ):
        refused = client.post("/data/remove", headers=headers, follow_redirects=False)
        assert refused.status_code == 403, headers
    assert "support_tickets.csv" in client.get("/data").text  # nothing was removed
    own = {"Sec-Fetch-Site": "same-origin", "Origin": HERE}
    assert client.post("/data/remove", headers=own, follow_redirects=False).status_code == 303
    sample(client)
    typed = {"Sec-Fetch-Site": "none"}  # the address bar, or a bookmark
    assert client.post("/data/remove", headers=typed, follow_redirects=False).status_code == 303
    sample(client)
    # No browser headers at all: a script, or a test client.
    assert client.post("/data/remove", follow_redirects=False).status_code == 303


def test_on_a_space_another_site_cannot_plant_its_token(monkeypatch, mocked_oauth):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with TestClient(create_app(SPACE, fake_services()[0]), base_url="https://testserver") as c:
        sample(c)
        planted = c.post(
            "/train/token",
            data={"token": "hf_attacker"},
            headers={"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"},
        )
        assert planted.status_code == 403
        assert "Sign in to train" in c.get("/train/review").text


def test_a_job_id_must_look_like_one(client, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("id,x\nTOPSECRET,1\n", encoding="utf-8")
    elsewhere = quote(str(tmp_path / "secret"), safe="")  # a drive and backslashes on Windows
    stolen = client.get(f"/runs/{elsewhere}/download.csv", follow_redirects=False)
    assert stolen.status_code == 404 and "TOPSECRET" not in stolen.text
    crafted = quote('job-b"><img src=x onerror=alert(1)>', safe="")
    for path in (
        "/models/{}/line",
        "/runs/{}/resume",
        "/runs/{}",
        "/runs/{}/status",
        "/predict/{}",
    ):
        answer = client.get(path.format(crafted), follow_redirects=False)
        assert answer.status_code == 404 and "<img" not in answer.text, path
    assert client.post(f"/runs/{crafted}/stop", follow_redirects=False).status_code == 404
    assert client.get("/runs/local", follow_redirects=False).status_code == 303  # no local run


def test_a_foreign_host_is_refused_on_this_computer(services_and_calls):
    from clef_tune.ui.server import bound_hosts

    services = services_and_calls[0]
    app = create_app(LOCAL, services, secret="test")
    with TestClient(app, base_url="http://rebound.example") as c:
        assert c.get("/").status_code == 400
        assert c.post("/data/sample", data={"name": "support_tickets"}).status_code == 400
    for url in ("http://localhost:7860", "http://127.0.0.1", "http://[::1]:7860"):
        with TestClient(app, base_url=url) as c:
            assert c.get("/").status_code == 200, url
    bound = create_app(LOCAL, services, secret="test", allowed_hosts=bound_hosts("192.168.1.5"))
    with TestClient(bound, base_url="http://192.168.1.5:7860") as c:
        assert c.get("/").status_code == 200
    with TestClient(bound, base_url="http://rebound.example") as c:
        assert c.get("/").status_code == 400


def test_serving_the_whole_network_warns_and_takes_any_host(
    services_and_calls, monkeypatch, capsys
):
    import uvicorn

    from clef_tune.ui import server

    served = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: served.update(app=app, **kwargs))
    server.launch(host="0.0.0.0", settings=LOCAL, services=services_and_calls[0])
    warning = capsys.readouterr().err
    assert "anyone on your network" in warning and "Hugging Face login" in warning
    with TestClient(served["app"], base_url="http://192.168.1.9:7860") as c:
        assert c.get("/").status_code == 200
    server.launch(settings=LOCAL, services=services_and_calls[0])
    assert capsys.readouterr().err == "" and served["host"] == "127.0.0.1"
    with TestClient(served["app"], base_url="http://192.168.1.9:7860") as c:
        assert c.get("/").status_code == 400


def test_slow_edits_run_off_the_event_loop(client, monkeypatch, tmp_path):
    from clef_tune.ui import handlers, runs, server

    seen = {}

    def watched(name, function):
        def call(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                seen[name] = "on the event loop"
            except RuntimeError:  # a worker thread has no running loop
                seen[name] = "in a worker thread"
            return function(*args, **kwargs)

        return call

    monkeypatch.setattr(handlers, "set_field", watched("field", handlers.set_field))
    monkeypatch.setattr(runs, "review_launch", watched("review", runs.review_launch))
    monkeypatch.setattr(server, "check_input", watched("rename", server.check_input))
    sample(client)
    client.post("/settings/field", data={"path": "train.epochs", "value": "2"}, headers=SCRIPT)
    client.post("/train/delete-data", data={"delete": ["", "1"]}, headers=SCRIPT)
    client.get("/predict/job-b")
    rows = [{"title": "s", "body": "b", "customer_tier": "pro"} for _ in range(3)]
    with open(write_csv(tmp_path / "renamed.csv", rows), "rb") as handle:
        client.post("/predict/job-b/upload", files={"file": ("renamed.csv", handle, "text/csv")})
    seen.pop("rename")  # the upload's check, which runs in the thread pool as well
    client.post("/predict/job-b/rename", data={"from": "title", "to": "subject"}, headers=SCRIPT)
    assert seen == dict.fromkeys(("field", "review", "rename"), "in a worker thread")
    kept = client.post("/train/delete-data", data={"delete": [""]}, headers=SCRIPT).text
    assert 'name="delete" value="1">' in kept
    deleted = client.post("/train/delete-data", data={"delete": ["", "1"]}, headers=SCRIPT).text
    assert 'name="delete" value="1" checked>' in deleted


# ---------------------------------------------------------------------------- signing in


def test_without_a_login_this_computer_says_how_to_log_in(logged_out):
    services, calls = fake_services()
    with local_client(services) as c:
        sample(c)
        review = c.get("/train/review").text
        models = c.get("/models").text
        started = c.post("/train/start", follow_redirects=False)
        pages = [review, models, c.get("/train").text, c.get("/runs/job-b").text]
        pages.append(c.get("/predict/job-b").text)
        pasted = c.post("/train/token", data={"token": "hf_pasted", "back": "/models"})
        assert pasted.url.path == "/models" and "run-b" in pasted.text
        assert "Signed in as <b>me</b>" in c.get("/train/review").text
    for page in (review, models):
        assert "hf auth login" in page and 'action="/train/token"' in page
    assert all("/oauth/" not in page for page in pages)
    assert started.status_code == 303 and started.headers["location"] == "/train"
    assert not [call for call in calls if call[0] == "submit_launch"]


def test_an_expired_sign_in_counts_as_signed_out(monkeypatch, mocked_oauth):
    import base64

    import huggingface_hub._oauth as oauth

    expired = {
        "access_token": "hf_expired",
        "token_type": "bearer",
        "expires_in": 8 * 3600,
        "scope": "openid profile",
        "expires_at": int((datetime.now() - timedelta(hours=1)).timestamp()),
        "userinfo": {"sub": "1", "name": "Dev", "preferred_username": "dev"},
    }
    monkeypatch.setattr(oauth, "_get_mocked_oauth_info", lambda: expired)
    monkeypatch.setenv("HF_TOKEN", "hf_server_owner")
    services, calls = fake_services()
    with TestClient(create_app(SPACE, services), base_url="https://testserver") as c:
        c.get("/oauth/huggingface/login?_target_url=/")
        assert ">Sign in with Hugging Face<" in c.get("/").text
        assert "Sign in with Hugging Face to see the models" in c.get("/models").text
        sample(c)
        assert "Sign in to train" in c.get("/train/review").text
        cookie = c.cookies.get("session")
        stored = json.loads(base64.b64decode(cookie.split(".")[0] + "=="))
    assert "oauth_info" not in stored
    assert not [call for call in calls if "hf_expired" in (*call[1], *call[2].values())]


def test_a_pasted_token_shows_its_own_account_and_switch_forgets_it(monkeypatch, mocked_oauth):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    services, _ = fake_services(whoami=lambda token: "owner" if token == "hf_pasted" else "me")
    with TestClient(create_app(SPACE, services), base_url="https://testserver") as c:
        c.get("/oauth/huggingface/login?_target_url=/")
        sample(c)
        c.post("/train/token", data={"token": "hf_pasted"})
        review = c.get("/train/review").text
        assert "Signed in as <b>owner</b>" in review and 'action="/train/switch"' in review
        assert '<span class="who">owner</span>' in c.get("/").text
        switched = c.post("/train/switch", follow_redirects=False)
        assert switched.headers["location"].startswith("/oauth/huggingface/logout")
        c.get(switched.headers["location"])
        assert "Sign in to train" in c.get("/train/review").text


def test_account_names_keep_only_answers_and_only_so_many():
    from clef_tune.ui.server import AccountNames

    def offline(token):
        raise OSError("offline")

    names = AccountNames(limit=2)
    assert names.lookup("hf_a", offline) is None
    assert names.lookup("hf_a", lambda token: "alice") == "alice"  # the failure wasn't kept
    assert names.lookup("hf_a", offline) == "alice"  # the answer was
    names.lookup("hf_b", lambda token: "bob")
    names.lookup("hf_c", lambda token: "carol")
    assert len(names) == 2


def test_a_failed_account_lookup_is_tried_again(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_machine")
    failures = iter([OSError("offline")])

    def whoami(token):
        for error in failures:
            raise error
        return "me"

    with local_client(fake_services(whoami=whoami)[0]) as c:
        assert "Hello again" not in c.get("/").text
        assert "Hello again" in c.get("/").text


def test_on_a_space_public_reads_send_no_token(monkeypatch, mocked_oauth, tmp_path):
    import huggingface_hub
    from tokenizers import Tokenizer, models, pre_tokenizers

    from clef_tune.prompt import get_token_counter
    from clef_tune.ui import services as services_module

    monkeypatch.setenv("HF_TOKEN", "hf_server_owner")
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    sent = []

    def download(repo_id, filename, **kwargs):
        sent.append(("tokenizer", kwargs.get("token")))
        return str(tmp_path / "tokenizer.json")

    def hardware(token=None):
        sent.append(("prices", token))
        return []

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "list_jobs_hardware", hardware)
    monkeypatch.setattr(services_module, "_prices", {"at": 0.0, "values": {}})
    get_token_counter.cache_clear()
    public = services_module.public_services()
    services, _ = fake_services(counter=public.counter, prices=public.prices)
    try:
        with TestClient(create_app(SPACE, services)) as c:
            sample(c)
            c.get("/settings")
    finally:
        get_token_counter.cache_clear()
    assert {"tokenizer", "prices"} <= {what for what, _ in sent}
    assert all(token is False for _, token in sent)


def test_a_space_uses_the_public_services_unless_told_otherwise(monkeypatch, mocked_oauth):
    from clef_tune.ui import server

    made = []
    monkeypatch.setattr(server, "public_services", lambda: made.append(1) or fake_services()[0])
    server.create_app(SPACE)
    assert made


# ---------------------------------------------------------------------------- lost and refused


def test_an_edit_after_the_setup_was_lost_starts_again(client):
    for path, data in (
        ("/settings/preset", {"preset": "quick"}),
        ("/settings/field", {"path": "train.epochs", "value": "2"}),
        ("/settings/name", {"name": "x"}),
        ("/settings/holdout", {"percent": "20"}),
        ("/data/role", {"column": "subject", "role": "skip"}),
        ("/questions/ask", {"qid": "urgency", "ask": "How soon?"}),
    ):
        response = client.post(path, data=data, headers=SCRIPT)
        assert response.status_code == 204 and response.headers["X-Redirect"] == "/data", path
    assert "This page has expired" in client.get("/data").text


def test_a_refused_role_change_says_why(client):
    sample(client)
    body = client.post("/data/role", data={"column": "subject", "role": "bogus"}, headers=SCRIPT)
    assert "Unknown role" in body.text


def test_an_oversized_upload_is_refused_before_it_is_read(client, monkeypatch):
    from starlette.requests import Request

    client.get("/predict/job-b")

    def unread(*args, **kwargs):
        raise AssertionError("the upload was read")

    monkeypatch.setattr(Request, "_get_form", unread)
    for path in ("/data/upload", "/predict/job-b/upload"):
        big = io.BytesIO(b"x" * (2 * 2**20))
        shown = client.post(path, files={"file": ("huge.csv", big, "text/csv")})
        assert "over the 1 MB limit" in shown.text, path


def test_an_upload_named_dot_dot_is_saved_under_a_safe_name(client):
    for name in ("..", ".", " "):
        shown = client.post(
            "/data/upload", files={"file": (name, io.BytesIO(b"a,b\n"), "text/csv")}
        )
        assert shown.status_code == 200 and "Could not use upload" in shown.text, name


def test_a_review_threshold_must_be_a_real_number(client, tmp_path):
    client.get("/predict/job-b")
    rows = [{"subject": "s", "body": "b", "customer_tier": "pro"} for _ in range(5)]
    with open(write_csv(tmp_path / "new.csv", rows), "rb") as handle:
        client.post("/predict/job-b/upload", files={"file": ("new.csv", handle, "text/csv")})
    for bad in ("nan", "inf", "-inf", "lots"):
        client.post("/predict/job-b/threshold", data={"percent": bad})
    assert 'name="percent" value="70"' in client.get("/predict/job-b").text
    client.post("/predict/job-b/threshold", data={"percent": "150"})
    assert 'name="percent" value="99"' in client.get("/predict/job-b").text


# ---------------------------------------------------------------------------- runs and predictions


def test_prediction_results_use_the_threshold_the_job_ran_with():
    def status(job_id, **kwargs):
        found = fakes.job_status(job_id)
        return dataclasses.replace(found, review_below=0.9) if job_id == "job-p" else found

    with local_client(fake_services(job_status=status)[0]) as c:
        c.get("/predict/job-b")
        c.post("/predict/job-b/threshold", data={"percent": "50"})
        page = c.get("/runs/job-p").text
    assert "less than 90% sure" in page


def test_a_run_says_how_long_it_should_take_from_its_own_estimate():
    def status(job_id, **kwargs):
        found = fakes.job_status(job_id)
        if job_id != "job-new":
            return found
        return dataclasses.replace(found, stage="SCHEDULING", progress=[], estimated_hours=0.5)

    with local_client(fake_services(job_status=status)[0]) as c:
        page = c.get("/runs/job-new").text
        polled = c.get("/runs/job-new/status").text
    for shown in (page, polled):
        assert "Expected to take about 30 minutes" in shown and "10 to 20" not in shown


def test_a_models_line_reads_its_run_once(client, services_and_calls):
    _, calls = services_and_calls
    assert "24 of 30" in client.get("/models/job-b/line").text
    assert len([call for call in calls if call[0] == "job_status"]) == 1


def test_results_that_are_not_readable_yet_are_tried_again():
    from clef_tune.hub import RunFiles
    from clef_tune.yamlio import load_yaml_file

    pulls = iter([OSError("not uploaded yet")] * 2)
    downloads = iter([OSError("still uploading")] * 2)

    def pull(repo_id, **kwargs):
        for error in pulls:
            raise error
        return RunFiles(fakes.recorded_metrics(), load_yaml_file(fakes.EXAMPLE_CONFIG), None)

    def download(job_id, out, **kwargs):
        for error in downloads:
            raise error
        return fakes._write(out, fakes.PREDICTIONS)

    services, _ = fake_services(pull_run_files=pull, download_predictions=download)
    with local_client(services) as c:
        for job_id, ready in (("job-b", "run-b is ready"), ("job-p", "3 tickets predicted")):
            page = c.get(f"/runs/{job_id}").text
            assert "readable yet" in page and f'data-poll="/runs/{job_id}/status"' in page
            waiting = c.get(f"/runs/{job_id}/status")
            assert "readable yet" in waiting.text and "X-Stop-Polling" not in waiting.headers
            done = c.get(f"/runs/{job_id}/status")
            assert done.headers["X-Stop-Polling"] == "1" and done.headers["X-Refresh"] == "1"
            assert ready in c.get(f"/runs/{job_id}").text


def test_a_quick_tests_results_say_it_barely_trained():
    from clef_tune.hub import RunFiles
    from clef_tune.yamlio import load_yaml_file

    def pull(repo_id, **kwargs):
        config = load_yaml_file(fakes.EXAMPLE_CONFIG) | {
            "train": {"mode": "head_only", "epochs": 1}
        }
        return RunFiles(fakes.recorded_metrics(), config, None)

    with local_client(fake_services(pull_run_files=pull)[0]) as c:
        page = c.get("/runs/job-b").text
    assert "This was a quick test, so it barely trained." in page


def test_deleting_a_runs_files_forgets_its_downloaded_spreadsheet(client, services_and_calls):
    _, calls = services_and_calls

    def downloads():
        return len([call for call in calls if call[0] == "download_predictions"])

    assert client.get("/runs/job-p/download.csv").status_code == 200
    assert client.get("/runs/job-p/download.xlsx").status_code == 200
    assert downloads() == 1
    deleted = client.post("/runs/job-p/delete-files", headers=SCRIPT).text
    assert "Deleted 1 file." in deleted
    client.get("/runs/job-p/download.csv")
    assert downloads() == 2


def test_the_calibration_sentence_needs_a_calibrated_model(tmp_path):
    from clef_tune.hub import RunFiles
    from clef_tune.yamlio import load_yaml_file

    def files(fitted):
        def pull(repo_id, **kwargs):
            metrics = fakes.recorded_metrics()
            if fitted is None:
                metrics.pop("calibration")
            else:
                for entry in metrics["calibration"].values():
                    entry["fitted"] = fitted
            return RunFiles(metrics, load_yaml_file(fakes.EXAMPLE_CONFIG), None)

        return pull

    rows = [{"subject": "s", "body": "b", "customer_tier": "pro"} for _ in range(5)]
    csv_path = write_csv(tmp_path / "new.csv", rows)
    for fitted in (True, False, None):
        with local_client(fake_services(pull_run_files=files(fitted))[0]) as c:
            c.get("/predict/job-b")
            with open(csv_path, "rb") as handle:
                c.post("/predict/job-b/upload", files={"file": ("new.csv", handle, "text/csv")})
            page = c.get("/predict/job-b").text
        assert ("was calibrated on your test" in page) is (fitted is True), fitted


def test_a_mapped_column_comes_back_under_the_persons_own_name(tmp_path):
    def status(job_id, **kwargs):
        found = fakes.job_status(job_id)
        names = {"ticket_id": "row", "subject": "title"}
        return dataclasses.replace(found, output_names=names) if job_id == "job-p" else found

    services, calls = fake_services(job_status=status)
    with local_client(services) as c:
        c.get("/predict/job-b")
        rows = [{"title": "s", "body": "b", "customer_tier": "pro"} for _ in range(3)]
        with open(write_csv(tmp_path / "renamed.csv", rows), "rb") as handle:
            c.post("/predict/job-b/upload", files={"file": ("renamed.csv", handle, "text/csv")})
        c.post("/predict/job-b/rename", data={"from": "title", "to": "subject"}, headers=SCRIPT)
        c.get("/predict/job-b/estimate")
        planned = next(call for call in calls if call[0] == "plan_prediction")[2]
        header = c.get("/runs/job-p/download.csv").text.lstrip("﻿").splitlines()[0]
        page = c.get("/runs/job-p").text
    assert planned["output_names"] == {"ticket_id": "row", "subject": "title"}
    assert header.startswith("row,title,body,customer_tier,department,")
    assert ">title</th>" in page and ">subject</th>" not in page


def test_the_stop_confirmation_says_what_it_stops(client):
    assert (
        '<button class="btn danger" type="submit">Stop training</button>'
        in client.get("/runs/job-a").text
    )
