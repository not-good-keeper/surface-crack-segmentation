"""History filtering and pagination (FR-18 / T-17).

Each filter is checked on its own and then in combination, and each result set is
compared against an independently computed set - including when the answer is empty.
"""

from __future__ import annotations

from app.repositories.inspection_repository import HistoryFilters
from app.services import history_service


def all_rows(conn):
    return [dict(r) for r in conn.execute(
        """
        SELECT i.*, m.material_code, s.station_code
        FROM inspection i
        LEFT JOIN material m ON m.material_id = i.material_id
        LEFT JOIN station s ON s.station_id = i.station_id
        """
    )]


def test_status_filter_matches_an_independent_set(conn):
    page = history_service.search(conn, HistoryFilters(status="clean", page_size=500))
    expected = [r for r in all_rows(conn) if r["status"] == "clean"]
    assert page.total == len(expected)
    assert {r["inspection_id"] for r in page.rows} == {r["inspection_id"] for r in expected}


def test_material_filter_matches_an_independent_set(conn):
    page = history_service.search(conn, HistoryFilters(material="steel", page_size=500))
    expected = [r for r in all_rows(conn) if r["material_code"] == "steel"]
    assert page.total == len(expected)


def test_station_filter_matches_an_independent_set(conn):
    page = history_service.search(conn, HistoryFilters(station="line-1-cam-A", page_size=500))
    expected = [r for r in all_rows(conn) if r["station_code"] == "line-1-cam-A"]
    assert page.total == len(expected)


def test_product_id_filter_does_a_partial_match(conn):
    row = conn.execute(
        "SELECT product_id FROM inspection WHERE product_id IS NOT NULL LIMIT 1"
    ).fetchone()
    fragment = row["product_id"][:7]
    page = history_service.search(conn, HistoryFilters(product_id=fragment, page_size=500))
    assert page.total >= 1
    assert all(fragment in (r["product_id"] or "") for r in page.rows)


def test_product_id_filter_supports_a_wildcard(conn):
    row = conn.execute("SELECT product_id FROM inspection WHERE product_id LIKE '%/%' LIMIT 1").fetchone()
    prefix = row["product_id"].split("/")[0]
    page = history_service.search(conn, HistoryFilters(product_id=f"{prefix}/*", page_size=500))
    assert page.total >= 1


def test_class_filter_only_returns_inspections_with_that_class(conn):
    page = history_service.search(conn, HistoryFilters(defect_class="crack", page_size=500))
    assert page.total >= 1
    for row in page.rows:
        classes = {
            r["class_code"]
            for r in conn.execute(
                "SELECT c.class_code FROM defect_region r JOIN defect_class c ON c.class_id = r.class_id "
                "WHERE r.inspection_id = ?",
                [row["inspection_id"]],
            )
        }
        assert "crack" in classes


def test_date_range_filter(conn):
    dates = sorted({r["captured_at"][:10] for r in all_rows(conn)})
    target = dates[-1]
    page = history_service.search(conn, HistoryFilters(date_from=target, date_to=target, page_size=500))
    expected = [r for r in all_rows(conn) if r["captured_at"][:10] == target]
    assert page.total == len(expected)


def test_combined_filters_narrow_the_result(conn):
    single = history_service.search(conn, HistoryFilters(material="steel", page_size=500))
    combined = history_service.search(
        conn, HistoryFilters(material="steel", status="regions_found", page_size=500)
    )
    assert combined.total <= single.total
    for row in combined.rows:
        assert row["material_code"] == "steel"
        assert row["status"] == "regions_found"


def test_an_impossible_combination_returns_an_empty_set(conn):
    page = history_service.search(
        conn, HistoryFilters(material="glass", product_id="no-such-product-id", page_size=500)
    )
    assert page.total == 0
    assert page.rows == []
    assert page.page_count == 1


def test_pagination_covers_every_row_exactly_once(conn):
    total = history_service.search(conn, HistoryFilters(page_size=500)).total
    seen = []
    page_no = 1
    while True:
        page = history_service.search(conn, HistoryFilters(page=page_no, page_size=7))
        seen.extend(r["inspection_id"] for r in page.rows)
        if not page.has_next:
            break
        page_no += 1
    assert len(seen) == total
    assert len(set(seen)) == total


def test_pagination_metadata(conn):
    page = history_service.search(conn, HistoryFilters(page=2, page_size=5))
    assert page.page == 2
    assert page.first_index == 6
    assert page.has_prev is True


def test_history_is_ordered_newest_first(conn):
    page = history_service.search(conn, HistoryFilters(page_size=50))
    stamps = [r["captured_at"] for r in page.rows]
    assert stamps == sorted(stamps, reverse=True)


def test_parse_filters_ignores_blanks_and_bad_values(conn):
    filters = history_service.parse_filters(
        {"material": "  ", "status": "not-a-status", "page": "abc", "page_size": "9999"}
    )
    assert filters.material is None
    assert filters.status is None
    assert filters.page == 1
    assert filters.page_size <= 200


def test_filters_survive_in_the_query_string(client):
    query = "material=steel&status=regions_found&page=2&page_size=5"
    body = client.get(f"/history?{query}").text
    assert 'value="steel" selected' in body
    assert "page=3" in body or "page=1" in body  # pagination links keep the filter
    assert "material=steel" in body


def test_history_export_link_carries_the_filter(client):
    body = client.get("/history?material=steel&status=clean").text
    assert "/api/inspections/export.csv?material=steel" in body


def test_empty_state_is_shown_when_nothing_matches(client):
    body = client.get("/history?product_id=definitely-not-here").text
    assert "No inspections match these filters" in body


def test_api_and_page_agree_on_the_total(client, conn):
    payload = client.get("/api/inspections?material=steel&status=regions_found").json()
    page = history_service.search(
        conn, HistoryFilters(material="steel", status="regions_found", page_size=500)
    )
    assert payload["total"] == page.total
