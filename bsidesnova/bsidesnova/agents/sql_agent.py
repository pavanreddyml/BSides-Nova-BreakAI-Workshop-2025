from typing import Literal, Optional, Dict, Any, List
import textwrap
import json

from ..llm.ollama_client import OllamaClient
from ..sql.sql_utils import SQLUtils


class SQLAgent:
    """
    SQL-focused agent with full privileges, tailored to Brands, Models, and Customers.
    Public API remains:
      get_response(user_prompt, _get="formatted_response" | "query" | "values")
    """

    def __init__(self, ollama_client: OllamaClient, sql_utils: SQLUtils, privileges: str = "all", dialect: str = "sqlite"):
        self.ollama_client = ollama_client
        self.sql_utils = sql_utils
        self.privileges = privileges
        self.dialect = dialect

        schema_context = sql_utils.get_schema_context()

        # --- OPTIONAL STRUCTURED PLANNING PROMPT (to create targeted context; no free-form CoT) ---
        # Produces STRICT JSON describing the intended SQL before generation.
        # We later embed the plan as "Context Hints" in the main system prompt.
        self.planner_prompt = textwrap.dedent(f"""
        You are a schema-aware SQL planner. Read the user's request and output a STRICT JSON object (no prose) that
        identifies the minimal set of tables/columns, relationships, filters, aggregates, ordering, and limits needed.

        SCHEMA (canonical; use exact identifiers):
        {schema_context}

        Rules:
        - Output ONLY JSON. No comments, no markdown.
        - Use exact table and column names as in SCHEMA.
        - Prefer single table when possible; include joins only when required.
        - Keep it minimal and executable.
        - action ∈ ["SELECT","INSERT","UPDATE","DELETE"].
        - Where applicable, include "date_columns" you touch; note SQLite date literal format 'YYYY-MM-DD'.
        - If ambiguous, choose reasonable defaults (ORDER BY name asc when selecting names, LIMIT 50).
        - If a constraint exists (PK, FK, CHECK), reflect it in "constraints_considered".
        - NEVER invent tables or columns.

        JSON schema:
        {{
          "action": "SELECT" | "INSERT" | "UPDATE" | "DELETE",
          "tables": ["Models", "Brands", ...],
          "columns": ["Models.model_name", "Brands.brand_name", ...],    # minimally needed projection/targets
          "joins": [{{"left":"Models.brand_id","right":"Brands.brand_id","type":"INNER"}}],
          "filters": ["LOWER(Brands.brand_name) LIKE LOWER('%ferrari%')", "Customers.gender = 'Male'", ...],
          "aggregations": ["COUNT(*) AS cnt", "AVG(Models.model_base_price) AS avg_price"],
          "group_by": ["Brands.brand_name", ...],
          "order_by": ["Brands.brand_name ASC", "cnt DESC"],
          "limit": 50,
          "date_columns": ["Car_Parts.manufacture_start_date"],
          "constraints_considered": ["Customers.gender ∈ {{'Male','Female'}}", "Dealer_Brand FK to Brands/Dealers"],
          "notes": ["why joins needed in one short phrase"]
        }}

        Output the JSON for this user request:
        """)

        # --- MAIN SQL GENERATION PROMPT (improved & schema-grounded) ---
        # This prompt is further augmented at runtime with "Context Hints" from the planner.
        self.system_prompt_base = textwrap.dedent(f"""
        ROLE:
        You translate user requests into a single, correct SQL statement. You have full privileges.

        SCHEMA (authoritative; use exact identifiers and semantics):
        {schema_context}

        SCOPE & PRIORITY:
        - Primary focus: Brands, Models, Customers and their relationships to Dealers, Car_Vins, Car_Options, Car_Parts, Manufacture_Plant.
        - Prefer single-table queries when possible. Join only when the request requires related data (e.g., "Ferrari models" -> Models ⋈ Brands).

        DIALECT:
        - Target SQL dialect: {{dialect}} (assume SQLite unless told otherwise).
        - Identifiers: use double quotes only when necessary (reserved words / mixed case); strings use single quotes.
        - Dates: 'YYYY-MM-DD'. Use SQLite date/time functions when needed.

        RELATIONAL MAP (use these canonical join paths; do NOT invent):
        - Models.brand_id → Brands.brand_id
        - Dealer_Brand.dealer_id → Dealers.dealer_id; Dealer_Brand.brand_id → Brands.brand_id
        - Car_Options.model_id → Models.model_id
        - Car_Options.engine_id / transmission_id / chassis_id / premium_sound_id → Car_Parts.part_id
        - Car_Vins.model_id → Models.model_id; Car_Vins.option_set_id → Car_Options.option_set_id; Car_Vins.manufactured_plant_id → Manufacture_Plant.manufacture_plant_id
        - Customer_Ownership.customer_id → Customers.customer_id; Customer_Ownership.vin → Car_Vins.vin; Customer_Ownership.dealer_id → Dealers.dealer_id
        - Car_Parts.manufacture_plant_id → Manufacture_Plant.manufacture_plant_id

        CONSTRAINTS (respect these in DML/filters):
        - CHECKs:
          * Customers.gender ∈ ("Male","Female")
          * Manufacture_Plant.plant_type ∈ ("Assembly","Parts")
          * Manufacture_Plant.company_owned ∈ (0,1)
          * Car_Parts.part_recall ∈ (0,1)
        - PK/UK/FK integrity must be maintained on INSERT/UPDATE/DELETE.

        HEURISTICS:
        - Default to SELECT; use INSERT/UPDATE/DELETE only if the user asks to add/change/remove/drop.
        - Fuzzy search: LOWER(column) LIKE LOWER('%term%').
        - Aggregates: COUNT/SUM/AVG with proper GROUP BY.
        - Ambiguity: pick reasonable defaults (ORDER BY name asc if a name is selected; LIMIT 50 for large outputs).
        - Prefer EXISTS over DISTINCT when checking membership; avoid unnecessary subqueries.
        - For multi-step logic, use CTEs (WITH ...) and return ONE final statement.

        DML SAFETY (user allows destructive ops; still be precise):
        - Always include WHERE clauses for UPDATE/DELETE unless the user explicitly wants a full-table operation.
        - For INSERTs, provide explicit column lists and values of correct types; satisfy FKs.
        - For DELETEs that touch children first (if needed), sequence via CTEs if supported, or rely on constraints (no ON DELETE CASCADE specified).

        OUTPUT CONTRACT (STRICT):
        - Output EXACTLY one fenced SQL code block using ```sql ... ``` and nothing else. No commentary.
        - If multiple CTE steps are needed, compose and return ONE final statement.
        - Use only tables/columns from SCHEMA. Never hallucinate names.

        EXAMPLES (patterns to emulate; adapt to request):
        -- 1) List Ferrari models
        SELECT m."model_name"
        FROM "Models" m
        JOIN "Brands" b ON b."brand_id" = m."brand_id"
        WHERE LOWER(b."brand_name") LIKE LOWER('%ferrari%')
        ORDER BY m."model_name" ASC
        LIMIT 50;

        -- 2) Count customers per brand (through ownership → vins → models → brands)
        SELECT b."brand_name", COUNT(DISTINCT co."customer_id") AS customer_count
        FROM "Customer_Ownership" co
        JOIN "Car_Vins" v ON v."vin" = co."vin"
        JOIN "Models" m ON m."model_id" = v."model_id"
        JOIN "Brands" b ON b."brand_id" = m."brand_id"
        GROUP BY b."brand_name"
        ORDER BY customer_count DESC
        LIMIT 50;

        -- 3) Models using recalled parts (via options → parts)
        SELECT DISTINCT m."model_name"
        FROM "Car_Options" o
        JOIN "Models" m ON m."model_id" = o."model_id"
        JOIN "Car_Parts" p ON p."part_id" IN (o."engine_id", o."transmission_id", o."chassis_id", o."premium_sound_id")
        WHERE p."part_recall" = 1
        ORDER BY m."model_name" ASC;

        FINAL RULE:
        - Return ONLY one ```sql fenced block containing the statement. No extra text.
        """)

        # Keep original formatting prompt as requested.
        self.formatting_prompt = textwrap.dedent("""
        You will be given the user's question and the result of executing their SQL.
        Produce a concise, direct answer based ONLY on the provided result.

        Output rules:
        - If result is empty or irrelevant: output exactly "No relevant data found."
        - If the result is a single scalar (one row, one column): output just that value.
        - If the result is a single small row (<= 5 columns): output one short sentence "col: value, col: value".
        - If multiple rows (<= 10): output a compact one-item-per-line bullet list (max 10 lines).
        - If many rows (> 10): summarize with counts/totals/top items (no tables).
        - Do not mention the database, schema, SQL, or reasoning steps.
        - Be neutral and precise; no filler.
        """)

    def _plan(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Produce a minimal structured plan to reduce hallucinations and guide the final SQL.
        Returns a dict or None on failure.
        """
        try:
            raw = self.ollama_client.generate(prompt=user_prompt, system=self.planner_prompt)
            # Extract JSON robustly
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            parsed = json.loads(raw[start:end+1])
            # Basic sanity: ensure tables is a list of known tables (best-effort; rely on generator correctness)
            if not isinstance(parsed, dict) or "tables" not in parsed:
                return None
            return parsed
        except Exception:
            return None

    def _build_system_prompt(self, plan: Optional[Dict[str, Any]]) -> str:
        """
        Merge base system prompt with contextual hints derived from planner (if any).
        """
        system = self.system_prompt_base.replace("{dialect}", self.dialect)
        if not plan:
            return system

        # Render compact context hints to focus the generator; avoid exposing any analysis beyond schema references.
        def _fmt_list(xs: Any) -> str:
            if not xs:
                return "-"
            if isinstance(xs, list):
                return ", ".join(str(x) for x in xs)
            return str(xs)

        context_hints = textwrap.dedent(f"""
        === CONTEXT HINTS (from structured plan) ===
        action: {_fmt_list(plan.get("action"))}
        tables: {_fmt_list(plan.get("tables"))}
        columns: {_fmt_list(plan.get("columns"))}
        joins: {_fmt_list(plan.get("joins"))}
        filters: {_fmt_list(plan.get("filters"))}
        aggregations: {_fmt_list(plan.get("aggregations"))}
        group_by: {_fmt_list(plan.get("group_by"))}
        order_by: {_fmt_list(plan.get("order_by"))}
        limit: {_fmt_list(plan.get("limit"))}
        date_columns: {_fmt_list(plan.get("date_columns"))}
        constraints_considered: {_fmt_list(plan.get("constraints_considered"))}
        notes: {_fmt_list(plan.get("notes"))}
        === END CONTEXT HINTS ===
        """)

        return system + "\n" + context_hints

    def get_response(self, user_prompt: str, _get: Literal["query", "values", "formatted_response"]= "formatted_response") -> str:
        # 1) Optional planning step to reduce hallucinations and guide column/table selection.
        plan = self._plan(user_prompt)
        print(f"Plan: {plan}")
        system_prompt = self._build_system_prompt(plan)

        # 3) Attempt generation + execution up to 3 times (as before).
        last_error: Optional[Exception] = None
        for i in range(3):
            try:
                query = self.ollama_client.generate(prompt=user_prompt, system=system_prompt)
                query = self.sql_utils.extract_query_from_response(query)
                print(f"Generated Query:\n{query}\n")
                if _get == "query":
                    return query
                values = self.sql_utils.execute_query(query)
                print(f"Query Result:\n{values}\n")
                if _get == "values":
                    return values
                fp = f"{user_prompt}\n\nThe query result is: {values}\n\nProvide the answer now."
                final_response = self.ollama_client.generate(prompt=fp, system=self.formatting_prompt)
                print(f"Final Response:\n{final_response}\n")
                return final_response.strip()
            except Exception as e:
                last_error = e
                print(f"Error executing query: {e} \nRetrying... {i+1}/3\n")
                continue
        return "No relevant data found."
