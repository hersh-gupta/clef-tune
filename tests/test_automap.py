from clef_tune.automap import Proposal, QuestionProposal, propose, propose_column


def rows(n, **columns):
    """columns: name -> function of the row index giving the cell text."""
    return [{name: make(i) for name, make in columns.items()} for i in range(n)]


def test_columns_are_proposed_from_their_values():
    data = rows(
        40,
        ticket_id=lambda i: f"T{i:03d}",
        subject=lambda i: f"subject {i}",
        dept=lambda i: ["billing", "billing", "technical", "account"][i % 4],
        outage=lambda i: ["Yes", "no"][i % 2],
        flag=lambda i: ["1", "0", ""][i % 3],
        notes=lambda i: " ",
    )
    assert propose(data, list(data[0])) == Proposal(
        id_column="ticket_id",
        state_columns=["subject"],
        questions=[
            QuestionProposal("dept", "dept", "choice", ["billing", "account", "technical"]),
            QuestionProposal("outage", "outage", "noul", ["true", "false"]),
            QuestionProposal("flag", "flag", "noul", ["true", "false"]),
        ],
        ignored=["notes"],
        native=False,
    )


def test_too_many_distinct_values_make_a_state_column():
    data = rows(120, code=lambda i: f"c{i % 51}", tier=lambda i: f"t{i % 50}")
    proposal = propose(data, ["code", "tier"])
    assert proposal.state_columns == ["code"]
    assert [q.id for q in proposal.questions] == ["tier"]


def test_mostly_unique_values_make_a_state_column():
    data = rows(6, size=lambda i: ["s", "m", "l", "xl", "s", "m"][i])  # 4 of 6 distinct
    assert propose(data, ["size"]).state_columns == ["size"]


def test_an_id_column_must_be_unique_and_complete():
    assert propose(rows(5, id=lambda i: f"r{i % 4}"), ["id"]).id_column is None
    gap = rows(5, row_id=lambda i: "" if i == 2 else f"r{i}")
    assert propose(gap, ["row_id"]).id_column is None
    assert propose(rows(5, Ticket_ID=lambda i: f"r{i}"), ["Ticket_ID"]).id_column == "Ticket_ID"


def test_a_single_column_can_be_proposed_on_demand():
    data = rows(10, size=lambda i: ["s", "m", "l"][i % 3])
    assert propose_column(data, "size") == QuestionProposal(
        "size", "size", "choice", ["s", "l", "m"]
    )  # by count (s: 4, l: 3, m: 3), then by text
    assert propose_column(rows(3, empty=lambda i: ""), "empty") is None


def test_native_rows_propose_questions_from_their_labels():
    data = [
        {"id": "a", "state": {"text": "x"}, "labels": {"dept": "billing", "spam": True}},
        {
            "id": "b",
            "state": {"text": "y"},
            "labels": {"dept": {"billing": 0.4, "technical": 0.6}},
        },
        {"id": "c", "state": {"text": "z"}, "labels": {"dept": "technical", "spam": "no"}},
    ]
    proposal = propose(data, ["id", "state", "labels"])
    assert proposal.native and proposal.state_columns == [] and proposal.id_column is None
    assert proposal.questions == [
        QuestionProposal("dept", "dept", "choice", ["billing", "technical"]),
        QuestionProposal("spam", "spam", "noul", ["true", "false"]),
    ]
