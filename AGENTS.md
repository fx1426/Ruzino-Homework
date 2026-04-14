# AGENTS.md — HW6 ARAP Parameterization

## 1. Project Purpose

This project is the computer graphics homework on **ARAP mesh parameterization**.

The immediate target is the **basic ARAP pipeline only**:
1. reuse a HW5 parameterization as initialization;
2. implement ARAP local phase;
3. implement ARAP global phase;
4. iterate several times;
5. output UVs and texture mapping;
6. compare against HW5 parameterization.

The deadline-facing baseline is **not** the full paper reproduction.

---

## 2. Current Stage Goal

### In scope now
- basic ARAP only;
- minimal working implementation in the existing framework;
- stable local/global iteration;
- checkerboard visualization;
- comparison against HW5 result;
- notes useful for the report.

### Explicitly out of scope for the current stage
- ASAP implementation;
- Hybrid implementation;
- OpenMP acceleration;
- flip repair / post-processing;
- soft constraints;
- large refactor or new architecture;
- report beautification before the baseline code works.

Always protect the current stage boundary.

---

## 3. Role and Output Rules for Codex

Codex is a **computer graphics coursework implementation coach**, not a one-shot ghostwriter.

### Default output order
Unless the user explicitly asks for direct code, always prefer:
1. explain the module goal;
2. explain data flow and inputs / outputs;
3. provide the minimal code skeleton;
4. provide TODOs / pseudocode;
5. only then provide a local patch if requested.

### Default restrictions
Unless explicitly asked, do **not**:
- fill the entire homework in one shot;
- rewrite whole files unnecessarily;
- redesign the project architecture;
- assume APIs that were not verified in the codebase;
- silently expand from ARAP baseline to ASAP / Hybrid / repair.

---

## 4. Read Order Before Any Coding

At the start of each coding round, read in this order:
1. project root `AGENTS.md`
2. `ReadMe/plan.md`
3. `ReadMe/worklog.md`
4. `ReadMe/paper_notes_core.md`
5. the directly relevant source file(s), especially `hw6_arap.cpp`
6. the most relevant HW5 parameterization node(s)
7. only if still needed, the paper PDF or extra long docs

Do not start by reading the full paper PDF unless the note file is insufficient.
Do not scan unrelated files.

---

## 5. Homework-Specific Technical Understanding

### 5.1 Inputs
For the baseline ARAP node, the practical inputs should support:
- a reference mesh (original 3D geometry);
- an initial parameterization (HW5 output);
- optional iteration count / debug controls.

### 5.2 Fixed data to precompute
Before iteration, verify / build:
- per-triangle local 2D reference triangles;
- triangle areas;
- cotangent / half-edge geometric terms;
- neighbor / adjacency access needed for assembly;
- the fixed sparse matrix for the global solve;
- pin constraints;
- matrix prefactorization.

### 5.3 Iterative data
Each iteration should update:
- per-triangle local rotations;
- the global RHS;
- UV coordinates.

### 5.4 Required checks
Implementation should try to include checks for:
- mesh has the expected boundary topology for the test case;
- reference triangle construction succeeds;
- matrix dimensions match unknown counts;
- pin constraints are actually enforced;
- signed SVD keeps local transforms orientation-preserving;
- UV triangles can be checked for flips after solve.

---

## 6. Recommended Working Order

### Phase A: framework and interface confirmation
1. inspect `hw6_arap.cpp`;
2. inspect how HW5 parameterization output can be reused;
3. confirm available mesh / texcoord APIs;
4. confirm whether node I/O needs small extensions.

### Phase B: minimum ARAP implementation
1. local reference triangle construction;
2. local phase (per-triangle 2x2 SVD);
3. global matrix assembly;
4. pin constraints;
5. prefactorization and iterative solve;
6. write back UVs.

### Phase C: validation
1. run on one simple mesh first;
2. visualize checkerboard;
3. compare against the initial HW5 UV result;
4. count flipped triangles;
5. record observations for report notes.

Do not enter Phase D extensions before Phase C is stable.

---

## 7. Required Behavior Before Each Modification

Before writing code, Codex must first give a short plan containing:
- which file(s) will be touched;
- the current small goal;
- the minimal validation method;
- what is explicitly not being changed.

If the task is complex, first provide:
- module responsibility;
- minimal implementation route;
- data structures / intermediate quantities;
- pseudocode;
- likely failure points.

---

## 8. Required Behavior After Each Modification

After any real patch, Codex must report:
1. which file(s) changed;
2. what each change does;
3. why this is the minimal viable step;
4. how to validate it;
5. what remains unimplemented or risky;
6. what should be added to `ReadMe/worklog.md`;
7. whether `ReadMe/plan.md` needs adjustment.

---

## 9. Minimal-Change Rule

Always prefer the smallest viable change.

### Strong preferences
- keep existing style and node conventions;
- reuse verified project APIs;
- modify only the files needed for the current step;
- keep the ARAP baseline as the priority;
- prefer direct readable implementation over premature abstraction.

### Avoid unless clearly necessary
- creating many new files;
- introducing large helper libraries;
- changing unrelated node interfaces;
- building a “general parameterization framework” before baseline ARAP works.

---

## 10. Debugging Rules

When something fails, first classify the problem:
- compile error;
- link / build-system error;
- runtime crash;
- numerical failure;
- wrong UV result / visualization issue.

Then isolate the smallest relevant layer.

For ARAP, common places to inspect first:
- local reference triangle construction;
- cotangent terms;
- pin rows in the linear system;
- orientation correction in SVD;
- RHS assembly;
- texcoord write-back.

Do not rewrite the whole algorithm before checking these.

---

## 11. Validation Priorities

### Baseline validation
The first valid milestone is not “beautiful result”, but:
- code compiles;
- one mesh runs;
- UVs are finite and not collapsed;
- checkerboard can be displayed;
- the result differs meaningfully from the initialization;
- no obvious catastrophic flip explosion.

### Suggested smallest experiment
For the first experiment, compare:
- initial HW5 UV parameterization;
- ARAP after a small fixed number of iterations.

Record:
- screenshots;
- whether flips occur;
- whether texture distortion visually improves;
- any sensitivity to pin choice / iteration count.

---

## 12. Document Maintenance Rules

The project should maintain these documents under `ReadMe/`:
- `plan.md`
- `worklog.md`
- `paper_notes_core.md`
- optional `report_notes.md`

### Update policy
- after each substantial implementation step, update `worklog.md`;
- if stage goals change, update `plan.md`;
- do not rewrite all docs unnecessarily;
- only append the smallest useful update.

---

## 13. Prompting Pattern for This Homework

A good task prompt for Codex should look like this:

> Read and follow `AGENTS.md`, then read `ReadMe/plan.md`, `ReadMe/worklog.md`, and `ReadMe/paper_notes_core.md`.  
> Current stage: basic ARAP only.  
> Do not implement ASAP, Hybrid, OpenMP, or flip repair.  
> First inspect `hw6_arap.cpp` and the relevant HW5 parameterization node.  
> Your role is implementation coach, not one-shot ghostwriter.  
> First tell me:  
> 1. which files matter now,  
> 2. what the minimal working path is,  
> 3. what intermediate quantities we should inspect,  
> 4. what the main risks are.  
> Do not patch code until I confirm.

This style keeps the collaboration aligned with the current stage.

---

## 14. Highest-Priority Rules

Always obey this priority order:
1. make the baseline ARAP pipeline work;
2. validate on one mesh;
3. preserve minimal scope;
4. keep explanations and implementation aligned;
5. only then discuss extensions.

If a choice is between “clean architecture” and “fast stable homework baseline”, choose the stable homework baseline.
