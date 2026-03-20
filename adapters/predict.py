"""MiroFish prediction adapter — HTTP client for the Flask API, with Claude fallback.

Three prediction modes (in priority order):
1. MiroFish + AutoResearch — full OASIS simulation with research-enhanced context
2. AutoResearch + Claude — research-enhanced direct estimation
3. Claude-only — fast direct estimation (fallback)
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass

import httpx

from adapters.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured output from prediction pipeline."""

    probability: float
    confidence: str  # "high", "medium", "low"
    reasoning: str
    report_text: str = ""
    session_id: str = ""
    source: str = "unknown"  # "mirofish", "research+claude", "claude"
    research_summary: str = ""


class PredictAdapter:
    """
    Prediction engine with three modes:
    1. MiroFish multi-agent simulation (when server is running)
    2. AutoResearch + Claude estimation (research-enhanced)
    3. Direct Claude estimation (fast fallback)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5001",
        llm: LLMClient | None = None,
        poll_interval: int = 5,
        poll_timeout: int = 90,
        use_research: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.llm = llm
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.use_research = use_research
        self._researcher = None
        self._mirofish_failed = False  # Skip MiroFish after first failure in a session

    @property
    def researcher(self):
        """Lazy-init the auto-researcher."""
        if self._researcher is None and self.llm and self.use_research:
            from trading.research import AutoResearcher
            self._researcher = AutoResearcher(self.llm)
        return self._researcher

    def is_mirofish_available(self) -> bool:
        """Check if MiroFish server is running and hasn't failed recently."""
        if self._mirofish_failed:
            return False
        try:
            with httpx.Client(base_url=self.base_url, timeout=5) as client:
                resp = client.get("/health")
                data = resp.json()
                return data.get("status") == "ok"
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return False

    def reset_mirofish(self):
        """Reset MiroFish failure flag (call between daemon cycles)."""
        self._mirofish_failed = False

    def run_prediction(
        self,
        market_question: str,
        market_data: dict | None = None,
        context: str = "",
    ) -> PredictionResult:
        """
        Run prediction pipeline.

        Priority: MiroFish → Research+Claude → Claude-only
        """
        # Gather research if available
        research_doc = ""
        research_summary = ""
        if self.researcher:
            try:
                logger.info("[PREDICT] Running auto-research...")
                report = self.researcher.research(market_question, market_data)
                research_doc = report.to_document()
                research_summary = report.summary
                logger.info(
                    f"[PREDICT] Research complete: {len(report.sources)} sources, "
                    f"base rate: {report.base_rate:.0%}"
                )
            except Exception as e:
                logger.warning(f"[PREDICT] Auto-research failed: {e}")

        # Try MiroFish with research context (skip if it already failed this session)
        if self.is_mirofish_available():
            try:
                logger.info("[PREDICT] MiroFish available — running simulation")
                combined_context = f"{context}\n\n{research_doc}" if research_doc else context
                result = self._run_mirofish_pipeline(market_question, combined_context)
                result.research_summary = research_summary
                return result
            except Exception as e:
                logger.warning(f"[PREDICT] MiroFish failed ({e}) — skipping for remaining markets this cycle")
                self._mirofish_failed = True  # Don't waste 5min on every market

        # Research-enhanced Claude estimation
        if research_doc and self.llm:
            try:
                logger.info("[PREDICT] Using research-enhanced Claude estimation")
                return self._research_enhanced_prediction(
                    market_question, market_data or {}, research_doc, research_summary
                )
            except Exception as e:
                logger.warning(f"[PREDICT] Research-enhanced prediction failed: {e}")

        # Pure Claude fallback
        return self._claude_prediction(market_question, market_data or {})

    def _research_enhanced_prediction(
        self, question: str, market_data: dict,
        research_doc: str, research_summary: str,
    ) -> PredictionResult:
        """Claude estimation enhanced with auto-research context."""
        if not self.llm:
            return self._claude_prediction(question, market_data)

        # Feed research as additional context
        enhanced_data = {**market_data, "research_context": research_doc}
        result = self.llm.estimate_probability(question, enhanced_data)

        return PredictionResult(
            probability=result["probability"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            source="research+claude",
            research_summary=research_summary,
        )

    def _claude_prediction(
        self, question: str, market_data: dict
    ) -> PredictionResult:
        """Direct Claude estimation — fast, no simulation."""
        if not self.llm:
            return PredictionResult(
                probability=0.5,
                confidence="low",
                reasoning="No prediction engine available",
                source="none",
            )

        result = self.llm.estimate_probability(question, market_data)
        return PredictionResult(
            probability=result["probability"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            source="claude",
        )

    def _run_mirofish_pipeline(self, question: str, context: str) -> PredictionResult:
        """Execute the full MiroFish multi-stage simulation pipeline.

        Flow:
        1. Upload research doc + generate ontology → project_id
        2. Build knowledge graph from ontology
        3. Create simulation on that project
        4. Prepare agents (generate personas)
        5. Start simulation (agents debate)
        6. Generate report from simulation
        7. Parse probability from report
        """
        http = httpx.Client(base_url=self.base_url, timeout=120)

        try:
            # Stage 1: Generate ontology from our research document
            seed_text = (
                f"Prediction Market Analysis:\n"
                f"Question: {question}\n\n"
                f"{context or 'Analyze using all available knowledge.'}\n\n"
                f"Task: Simulate public discourse and determine the probability "
                f"of this event occurring."
            )

            # Create a virtual file for the ontology generator
            files = {
                "files": ("research.txt", io.BytesIO(seed_text.encode()), "text/plain"),
            }
            data = {
                "simulation_requirement": question,
                "project_name": f"polymarket-{int(time.time())}",
            }

            logger.info("  [1/6] Generating ontology...")
            ontology_resp = http.post(
                "/api/graph/ontology/generate", files=files, data=data
            )
            ontology_resp.raise_for_status()
            ontology_data = ontology_resp.json()

            if not ontology_data.get("success"):
                raise RuntimeError(f"Ontology generation failed: {ontology_data.get('error')}")

            project_id = ontology_data["data"]["project_id"]
            logger.info(f"  [1/6] Project created: {project_id}")

            # Stage 2: Build knowledge graph
            logger.info("  [2/6] Building knowledge graph...")
            build_resp = http.post("/api/graph/build", json={
                "project_id": project_id,
            })
            build_resp.raise_for_status()
            build_data = build_resp.json()

            if not build_data.get("success"):
                raise RuntimeError(f"Graph build failed: {build_data.get('error')}")

            task_id = build_data["data"].get("task_id")
            if task_id:
                self._poll_task(http, task_id, "graph build")

            # Stage 3: Create simulation
            logger.info("  [3/6] Creating simulation...")
            sim_resp = http.post("/api/simulation/create", json={
                "project_id": project_id,
                "enable_twitter": True,
                "enable_reddit": True,
            })
            sim_resp.raise_for_status()
            sim_data = sim_resp.json()

            if not sim_data.get("success"):
                raise RuntimeError(f"Simulation create failed: {sim_data.get('error')}")

            sim_id = sim_data["data"]["simulation_id"]

            # Stage 4: Prepare (generate agent personas)
            logger.info("  [4/6] Preparing agents...")
            prep_resp = http.post("/api/simulation/prepare", json={
                "simulation_id": sim_id,
            })
            prep_resp.raise_for_status()

            # Poll preparation status
            self._poll_simulation_status(http, sim_id, "prepare")

            # Stage 5: Start simulation
            logger.info("  [5/6] Running OASIS simulation...")
            start_resp = http.post("/api/simulation/start", json={
                "simulation_id": sim_id,
            })
            start_resp.raise_for_status()

            # Poll run status
            self._poll_simulation_status(http, sim_id, "run")

            # Stage 6: Generate report
            logger.info("  [6/6] Generating prediction report...")
            report_resp = http.post("/api/report/generate", json={
                "simulation_id": sim_id,
            })
            report_resp.raise_for_status()
            report_data = report_resp.json()

            if not report_data.get("success"):
                raise RuntimeError(f"Report generation failed: {report_data.get('error')}")

            report_id = report_data["data"].get("report_id")

            # Poll report progress
            if report_id:
                self._poll_report_progress(http, report_id)

            # Fetch final report
            final_resp = http.get(f"/api/report/{report_id}")
            final_resp.raise_for_status()
            final_data = final_resp.json()
            report_text = ""
            if final_data.get("success"):
                report_obj = final_data.get("data", {})
                report_text = report_obj.get("content", "")
                if not report_text:
                    # Try sections
                    sections = report_obj.get("sections", [])
                    report_text = "\n\n".join(
                        s.get("content", "") for s in sections
                    )

            logger.info(f"  Report generated: {len(report_text)} chars")

            # Parse probability from report
            if self.llm and report_text:
                result = self.llm.extract_probability(report_text, question)
                return PredictionResult(
                    probability=result["probability"],
                    confidence=result["confidence"],
                    reasoning=result["reasoning"],
                    report_text=report_text,
                    session_id=sim_id,
                    source="mirofish",
                )

            return PredictionResult(
                probability=0.5,
                confidence="low",
                reasoning="Report generated but could not extract probability",
                report_text=report_text,
                session_id=sim_id,
                source="mirofish",
            )
        finally:
            http.close()

    def _poll_task(self, http: httpx.Client, task_id: str, label: str):
        """Poll a graph build task until done."""
        elapsed = 0
        while elapsed < self.poll_timeout:
            try:
                resp = http.get(f"/api/graph/task/{task_id}")
                data = resp.json()
                if data.get("success"):
                    task = data.get("data", {})
                    status = task.get("status", "").lower()
                    if status in ("completed", "done", "success"):
                        logger.info(f"  {label} completed")
                        return
                    if status in ("failed", "error"):
                        raise RuntimeError(f"{label} failed: {task.get('error')}")
            except httpx.HTTPError:
                pass

            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

        raise TimeoutError(f"{label} timed out after {self.poll_timeout}s")

    def _poll_simulation_status(self, http: httpx.Client, sim_id: str, phase: str):
        """Poll simulation run-status until phase completes."""
        elapsed = 0
        while elapsed < self.poll_timeout:
            try:
                resp = http.get(f"/api/simulation/{sim_id}/run-status")
                data = resp.json()
                if data.get("success"):
                    status_data = data.get("data", {})
                    status = status_data.get("status", "").lower()

                    if phase == "prepare":
                        if status in ("prepared", "ready", "idle", "created"):
                            # Check prepare status endpoint
                            prep_resp = http.post(
                                "/api/simulation/prepare/status",
                                json={"simulation_id": sim_id},
                            )
                            prep_data = prep_resp.json()
                            if prep_data.get("success"):
                                prep_status = prep_data.get("data", {}).get("status", "")
                                if prep_status in ("completed", "done"):
                                    logger.info(f"  Preparation completed")
                                    return
                    elif phase == "run":
                        if status in ("completed", "done", "finished"):
                            logger.info(f"  Simulation completed")
                            return
                        if status in ("failed", "error"):
                            raise RuntimeError(
                                f"Simulation failed: {status_data.get('error')}"
                            )
            except httpx.HTTPError:
                pass

            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

        raise TimeoutError(f"Simulation {phase} timed out after {self.poll_timeout}s")

    def _poll_report_progress(self, http: httpx.Client, report_id: str):
        """Poll report generation progress."""
        elapsed = 0
        while elapsed < self.poll_timeout:
            try:
                resp = http.get(f"/api/report/{report_id}/progress")
                data = resp.json()
                if data.get("success"):
                    progress = data.get("data", {})
                    status = progress.get("status", "").lower()
                    pct = progress.get("progress", 0)
                    if status in ("completed", "done"):
                        logger.info(f"  Report generation completed")
                        return
                    if status in ("failed", "error"):
                        raise RuntimeError(f"Report failed: {progress.get('error')}")
                    logger.debug(f"  Report progress: {pct}%")
            except httpx.HTTPError:
                pass

            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

        raise TimeoutError(f"Report generation timed out after {self.poll_timeout}s")
