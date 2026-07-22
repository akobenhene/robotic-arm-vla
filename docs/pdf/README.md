# PDF project reports

Well-formatted print reports with **Figma/FigJam frameworks embedded**.

| File | Contents |
|------|----------|
| [Full_Project_Report.pdf](Full_Project_Report.pdf) | Abstract, intro, Grid Dynamics relevance, system design, results, soft-sensor link |
| [Grid_Dynamics_Relevance_Brief.pdf](Grid_Dynamics_Relevance_Brief.pdf) | Screening brief with architecture + GAIN loop figures |
| [Frameworks_Atlas.pdf](Frameworks_Atlas.pdf) | All six Figma frameworks with captions |

**Source board:** https://www.figma.com/board/CmvFbnixCtXsehlEUMbEnZ  
**Figures:** `docs/figures/*.png` (exported from FigJam sections)

Regenerate after updating diagrams:

```powershell
.\.venv\Scripts\python.exe scripts\generate_pdf_reports.py
```
