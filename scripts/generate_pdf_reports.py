"""Generate formatted PDF project reports with embedded Figma frameworks."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "pdf"
BOARD = "https://www.figma.com/board/CmvFbnixCtXsehlEUMbEnZ"
REPO = "https://github.com/akobenhene/robotic-arm-vla"

FIGURES = [
    ("01_system_architecture.png", "Figure 1. Physical AI system architecture (Figma)."),
    ("02_gain_delivery_loop.png", "Figure 2. Grid Dynamics GAIN delivery loop (Figma)."),
    ("03_closed_loop_control.png", "Figure 3. Closed-loop visuomotor control framework (Figma)."),
    ("04_evaluation_ablation.png", "Figure 4. Evaluation and ablation framework (Figma)."),
    ("05_soft_sensor_mapping.png", "Figure 5. Soft-sensor research mapped to Physical AI (Figma)."),
    ("06_reward_state_machine.png", "Figure 6. TransferCube sparse-reward state machine (Figma)."),
]


class ReportPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, self.report_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200, 200, 200)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}  |  {REPO}", align="C")

    def cover(self, subtitle: str, meta_lines: list[str]) -> None:
        self.add_page()
        self.set_y(40)
        self.set_font("Helvetica", "B", 20)
        self.multi_cell(0, 10, self.report_title, align="C")
        self.ln(6)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 7, subtitle, align="C")
        self.ln(12)
        y = self.get_y()
        self.set_draw_color(30, 90, 140)
        self.set_line_width(0.6)
        self.line(55, y, 155, y)
        self.set_y(y + 12)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        for line in meta_lines:
            self.multi_cell(0, 6, line, align="C")
            self.ln(1)
        self.ln(8)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, f"Framework source: {BOARD}", align="C")
        self.set_text_color(0, 0, 0)

    def h1(self, text: str) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 55, 95)
        self.multi_cell(0, 8, text)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def h2(self, text: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(35, 35, 35)
        self.multi_cell(0, 7, text)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_x(22)
        self.multi_cell(0, 5.5, f"- {text}")

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[float]) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 238, 246)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        for row in rows:
            # estimate height from longest wrapped cell
            line_counts = [
                max(1, len(self.multi_cell(w, 4.5, c, dry_run=True, output="LINES")))
                for c, w in zip(row, col_widths)
            ]
            h = 4.5 * max(line_counts) + 2
            if self.get_y() + h > self.h - 20:
                self.add_page()
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(230, 238, 246)
                for hh, w in zip(headers, col_widths):
                    self.cell(w, 7, hh, border=1, fill=True)
                self.ln()
                self.set_font("Helvetica", "", 8.5)
            x0 = self.get_x()
            y0 = self.get_y()
            for c, w in zip(row, col_widths):
                self.rect(self.get_x(), y0, w, h)
                self.multi_cell(w, 4.5, c)
                self.set_xy(x0 := x0 + w, y0)
            self.set_xy(18, y0 + h)
        self.ln(3)

    def figure(self, filename: str, caption: str, max_h: float = 120) -> None:
        path = FIG / filename
        if not path.exists():
            self.body(f"[Missing figure: {filename}]")
            return
        self.ln(2)
        if self.get_y() > 200:
            self.add_page()
        # keep room for caption
        avail_w = 174
        avail_h = min(max_h, self.h - self.get_y() - 28)
        self.image(str(path), w=avail_w, h=avail_h, keep_aspect_ratio=True)
        self.ln(2)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, caption)
        self.set_text_color(0, 0, 0)
        self.ln(3)


def write_full_report(path: Path) -> None:
    pdf = ReportPDF("Full Project Report - Robotic Arm VLA")
    pdf.alias_nb_pages()
    pdf.cover(
        "Text-Conditioned Robotic Manipulation via Vision-Language-Action Policies",
        [
            "Physical AI portfolio report for Grid Dynamics review",
            "Stack: Python 3.11 / PyTorch / MuJoCo / gym-aloha / Hugging Face LeRobot",
            "Release: v1.0.0",
            "Repository: github.com/akobenhene/robotic-arm-vla",
        ],
    )

    pdf.add_page()
    pdf.h1("1. Abstract")
    pdf.body(
        "This report presents an end-to-end Physical AI system for tabletop robotic "
        "manipulation. A bimanual Aloha robot in MuJoCo observes RGB imagery and "
        "proprioceptive state, optionally conditioned on natural language, and outputs "
        "continuous motor commands in a closed loop. Two LeRobot policies are compared: "
        "ACT, which solves TransferCube, and SmolVLA, which demonstrably changes actions "
        "when the prompt changes. Multi-seed evaluation yields 50% ACT success on CPU "
        "(25/50). The delivery includes CI, release artifacts, finetune and SO-100 docs, "
        "and Figma framework diagrams embedded below."
    )

    pdf.h1("2. Introduction")
    pdf.body(
        "Industrial automation is moving from fixed trajectories toward perception-driven, "
        "learning-based control. Clients ask for systems that see with cameras, optionally "
        "follow language work orders, validate safely in simulation, and improve from "
        "demonstrations. This project implements that loop on Aloha TransferCube: pick a "
        "red cube with the right arm and transfer it so the left arm holds it off the table "
        "(success when sparse reward reaches 4)."
    )
    pdf.body(
        "Contributions: closed-loop visuomotor control; multimodal VLA interface; "
        "quantitative multi-seed evaluation; comparative ablation narrative; and "
        "production-minded packaging suitable for Forward Deployed Engineering review."
    )

    pdf.h1("3. Relevance for Grid Dynamics")
    pdf.body(
        "Grid Dynamics invests in Physical AI / GAIN, digital twins, IoT and edge robotics, "
        "NVIDIA partnerships, and Forward Deployed Engineering. This project is a portable "
        "demonstration of the same delivery pattern: simulate, measure KPIs, refine "
        "imitation or VLA models, package for stakeholders, then plan hardware transfer."
    )
    pdf.figure("02_gain_delivery_loop.png", FIGURES[1][1], max_h=55)
    pdf.h2("3.1 Capability alignment")
    pdf.table(
        ["Grid Dynamics theme", "Evidence in this project"],
        [
            ["Robotic manipulation", "Closed-loop ACT on bimanual TransferCube"],
            ["Validate before hardware", "MuJoCo twin + 50-seed success metrics"],
            ["Imitation / foundation refine", "Hub load + SmolVLA finetune playbook"],
            ["Instruction-driven workflows", "Language-conditioned SmolVLA ablation"],
            ["Production engineering", "CI smoke, release, PDF/MP4 artifacts"],
            ["FDE ownership model", "Env adapter to policy to KPI to docs"],
        ],
        [70, 104],
    )
    pdf.h2("3.2 Client use")
    for b in [
        "PoC sprint: matched sim task, Hub policy, multi-seed eval in days.",
        "Stakeholder pack: MP4 for operators, JSON rates for CTOs.",
        "Modular policy swap: ACT today, finetuned SmolVLA tomorrow.",
        "Soft-sensing cross-sell: same multimodal stack for vision QC.",
    ]:
        pdf.bullet(b)

    pdf.h1("4. System design")
    pdf.body(
        "The control loop is closed: each action changes the next image and state. "
        "RoboticsEnvWrapper converts gym-aloha observations into LeRobot batches. "
        "policy.py exposes ACT, SmolVLA, and a Mock backend behind predict()."
    )
    pdf.figure("01_system_architecture.png", FIGURES[0][1], max_h=105)
    pdf.figure("03_closed_loop_control.png", FIGURES[2][1], max_h=70)

    pdf.h1("5. Task and reward ladder")
    pdf.body(
        "TransferCube uses a sparse staged reward. Industrial KPIs are often similarly "
        "sparse (cycle complete / fail), so staged debugging matters in the field."
    )
    pdf.figure("06_reward_state_machine.png", FIGURES[5][1], max_h=115)
    pdf.table(
        ["Reward", "Meaning"],
        [
            ["0", "No useful contact"],
            ["1", "Right gripper touches cube"],
            ["2", "Right lifts cube off table"],
            ["3", "Left gripper contacts cube"],
            ["4", "Left holds cube off table (SUCCESS)"],
        ],
        [30, 144],
    )

    pdf.h1("6. Evaluation framework")
    pdf.body(
        "Task success and language sensitivity are measured separately. ACT is the task "
        "expert; SmolVLA proves the instruction interface. Conflating the two would "
        "mislead stakeholders."
    )
    pdf.figure("04_evaluation_ablation.png", FIGURES[3][1], max_h=130)
    pdf.h2("6.1 Results")
    pdf.table(
        ["Metric", "Value"],
        [
            ["ACT seeds evaluated", "0-49 (50 episodes)"],
            ["ACT success rate (CPU)", "25/50 = 50.0%"],
            ["Hero seed", "36"],
            ["Hub GPU reference", "~83% / 500 episodes"],
            ["SmolVLA language sensitive", "true (L1 action delta ~0.03)"],
            ["SmolVLA full-task success", "Not reliable on this checkpoint"],
        ],
        [70, 104],
    )

    pdf.h1("7. Soft sensing and industrial estimation")
    pdf.body(
        "AI soft sensors infer hard-to-measure quantities from cheap sensors and models. "
        "This robotics stack is the control counterpart: infer next actions from RGB and "
        "proprioception under physics constraints, then score with a KPI."
    )
    pdf.figure("05_soft_sensor_mapping.png", FIGURES[4][1], max_h=60)

    pdf.h1("8. Engineering lessons")
    for b in [
        "Zero-action warm-up after reset caused distribution shift and killed ACT success.",
        "Hub normalize buffers must be applied explicitly under LeRobot 0.4.x.",
        "Embodiment must match pretrained weights (Aloha, not FetchReach).",
        "Language sensitivity is not the same claim as task competence.",
        "Artifacts (GIF/MP4 + JSON + seeds) beat anecdotal demos.",
    ]:
        pdf.bullet(b)

    pdf.h1("9. Limitations and next steps")
    pdf.table(
        ["Limitation", "Next step"],
        [
            ["SmolVLA weak on full transfer", "GPU finetune on human TransferCube data"],
            ["CPU eval below Hub 83%", "GPU eval; more seeds"],
            ["Sim only", "SO-100 record, finetune, deploy path"],
            ["Single task", "Add insertion / pick-place with matched weights"],
        ],
        [75, 99],
    )

    pdf.h1("10. Conclusions")
    pdf.body(
        "This project is a miniature Physical AI delivery: perception, multimodal policy, "
        "closed-loop control, quantitative evaluation, CI, and documentation. For Grid "
        "Dynamics it evidences readiness for GAIN-style engagements: digital-twin "
        "validation, imitation/VLA integration, KPI-driven demos, and honest sim-to-real "
        "scoping without overclaiming Omniverse fidelity or zero-shot hardware transfer."
    )

    pdf.h1("11. Reproduce")
    pdf.body(
        "PowerShell (use the project .venv Python 3.11):\n"
        ".\\.venv\\Scripts\\python.exe main.py --policy act --seed 36\n"
        ".\\.venv\\Scripts\\python.exe prompt_ablation.py\n"
        ".\\.venv\\Scripts\\python.exe compare_policies.py --seed 36\n"
        ".\\.venv\\Scripts\\python.exe evaluate.py --policy act --seeds 0-49 "
        "--continue-after-success"
    )

    pdf.output(path)


def write_grid_brief(path: Path) -> None:
    pdf = ReportPDF("Grid Dynamics Relevance Brief")
    pdf.alias_nb_pages()
    pdf.cover(
        "How this Physical AI demo maps to GAIN, digital twins, and FDE delivery",
        [
            "One-pager extended brief for screening and stakeholder review",
            "Companion to Full_Project_Report.pdf",
            "FigJam frameworks embedded from project board",
        ],
    )
    pdf.add_page()
    pdf.h1("Positioning statement")
    pdf.body(
        "I built an end-to-end Physical AI pipeline: MuJoCo Aloha simulation, Hugging Face "
        "LeRobot ACT for successful bimanual cube transfer, and SmolVLA for "
        "language-conditioned actions. I evaluated 50 seeds at 50% CPU success, packaged "
        "CI and a release, and documented finetune and SO-100 paths. It follows the same "
        "simulate, measure, refine loop Grid Dynamics uses in GAIN Physical AI work."
    )

    pdf.h1("Delivery loop")
    pdf.figure("02_gain_delivery_loop.png", FIGURES[1][1], max_h=55)

    pdf.h1("Technical stack at a glance")
    pdf.figure("01_system_architecture.png", FIGURES[0][1], max_h=100)

    pdf.h1("What reviewers should look at")
    pdf.table(
        ["Artifact", "Why it matters"],
        [
            ["demo_output.mp4", "ACT solves TransferCube (seed 36)"],
            ["comparison_act_vs_smolvla.mp4", "Reliability vs language interface"],
            ["eval_results.json", "50% multi-seed KPI, not one lucky GIF"],
            ["prompt_ablation.json", "language_sensitive = true"],
            ["FigJam board", "Editable architecture frameworks"],
            ["docs/FULL_REPORT.md", "Full narrative + soft-sensor bridge"],
        ],
        [70, 104],
    )

    pdf.h1("Honest boundaries")
    for b in [
        "Not Omniverse-scale plant twin fidelity.",
        "Not claiming Aloha ACT zero-shots to SO-100.",
        "Not claiming laptop CPU matches Hub ~83% GPU eval.",
        "SmolVLA success needs finetune; language proof is separate.",
    ]:
        pdf.bullet(b)

    pdf.h1("Soft-sensor bridge")
    pdf.figure("05_soft_sensor_mapping.png", FIGURES[4][1], max_h=55)
    pdf.body(
        "Positions the author across perception, estimation, and action: soft sensing "
        "for visibility, Physical AI for actuation, digital twins for safe validation."
    )
    pdf.output(path)


def write_frameworks_atlas(path: Path) -> None:
    pdf = ReportPDF("Frameworks Atlas - Figma Diagrams")
    pdf.alias_nb_pages()
    pdf.cover(
        "All project frameworks exported from FigJam and inserted for print review",
        [
            f"Source board: {BOARD}",
            "Use with Full_Project_Report.pdf and Grid_Dynamics_Relevance_Brief.pdf",
        ],
    )
    captions = {
        "01_system_architecture.png": (
            "System architecture",
            "Operators run main.py; services wrap the env and policies; artifacts and "
            "external Hub/MuJoCo platforms complete the stack.",
        ),
        "02_gain_delivery_loop.png": (
            "Grid Dynamics GAIN delivery loop",
            "Discovery through simulation, KPI review, policy refine, packaging, and "
            "FDE hardware planning.",
        ),
        "03_closed_loop_control.png": (
            "Closed-loop control",
            "Sense RGB and state, optional language, policy action, MuJoCo step, "
            "success check at reward 4.",
        ),
        "04_evaluation_ablation.png": (
            "Evaluation and ablation",
            "ACT success-rate track versus SmolVLA language-sensitivity track, merged "
            "into comparative reporting.",
        ),
        "05_soft_sensor_mapping.png": (
            "Soft sensor mapping",
            "Maps process soft-sensing research ideas onto this robotics control stack "
            "for Grid Dynamics positioning.",
        ),
        "06_reward_state_machine.png": (
            "TransferCube reward states",
            "Sparse ladder from no contact (R0) to left-arm hold success (R4).",
        ),
    }
    for fname, default_caption in FIGURES:
        title, blurb = captions[fname]
        pdf.add_page()
        pdf.h1(title)
        pdf.body(blurb)
        max_h = 150 if "architecture" in fname or "evaluation" in fname or "reward" in fname else 90
        pdf.figure(fname, default_caption, max_h=max_h)
    pdf.output(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [n for n, _ in FIGURES if not (FIG / n).exists()]
    if missing:
        raise SystemExit(f"Missing figures: {missing}")

    full = OUT / "Full_Project_Report.pdf"
    brief = OUT / "Grid_Dynamics_Relevance_Brief.pdf"
    atlas = OUT / "Frameworks_Atlas.pdf"
    write_full_report(full)
    write_grid_brief(brief)
    write_frameworks_atlas(atlas)
    for p in (full, brief, atlas):
        print(f"Wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
