# The GompaTech Bridge
### Content architecture & creative direction for gompatech.com

This is the strategy layer — narrative, information architecture, section-by-section copy, and a design system spec that bridges `teamcenter.html` (→ `/plm`) and `foundry.html` (→ `/foundry`). No HTML/CSS here by design; that's the next iteration. Everything below is meant to be handed to whoever builds the page — including a future Claude session — as a working brief.

---

## 1. The bridge, in one line

Both service lines sell the same underlying fix, applied to two different layers of a manufacturer's stack: **closing the gap between the process that was defined and the process that's actually happening.** The PLM/Teamcenter practice closes that gap inside engineering systems. The computer-vision line closes it on the shop floor.

The word for that gap already exists in both worlds — you don't have to invent a metaphor to force this merger. Engineering teams file *deviations* (ECRs, non-conformances) when the system stops matching intent. Foundries live and die by *SOP deviations* — it's literally the word `foundry.html` already uses ("Your process deviations don't announce themselves"). That's not a coincidence worth papering over — it's the actual connective tissue, and it's specific to your two businesses rather than a generic pitch that could describe any consultancy.

That gives a trilogy that fits the voice both pages already use — a two-beat setup and a "Zero X" punchline — without inventing a new pattern:

| Level | Line | Status |
|---|---|---|
| **gompatech.com (master)** | **Define the Process. Defend the Process. Zero Deviation.** | New — proposed |
| `/plm` — PLM & Teamcenter | Stabilize Teamcenter. Accelerate Engineering. Zero Downtime. | Existing — unchanged |
| `/foundry` — Foundry Intelligence | Melt Fast. Pour Fast. Zero Defects. | Existing — unchanged |

"Zero Downtime" and "Zero Defects" turn out to be two symptoms of the same disease. "Zero Deviation" is the diagnosis. That's the whole thesis — everything below just builds it out into pages, sections, and copy.

---

## 2. Brand architecture — how the three pages relate

### Naming hierarchy
Right now the two pages use the name inconsistently — `teamcenter.html`'s nav spells out **GOMPA TECHNOLOGIES** in full caps mono; `foundry.html`'s nav and footer use the **Gompa`Tech`** lockup (with the accent-colored second half). Standardize on:

- **Legal/formal name:** Gompa Technologies — reserve for copyright lines and anywhere formality is expected.
- **Wordmark:** Gompa**Tech** — adopt `foundry.html`'s existing lockup everywhere, including retrofitting it onto `/plm`'s nav. One mark, three pages.
- **Practice labels** (not separate legal entities, just nav/card language): **PLM & Teamcenter Advisory** and **Manufacturing Process Intelligence** (with Foundry Intelligence as its current flagship instance — see §6).

### The hub's real job
`teamcenter.html` and `foundry.html` are each already linked from a cold outreach email or a WhatsApp share respectively — most first-time visitors to either page never touch the homepage at all. So gompatech.com is mostly a **second visit**: someone cross-checking "is this a real outfit?", a returning contact deciding whether the other service line is relevant to them too, or a prospect who found you cold (search, LinkedIn, referral) with no idea yet which door they want. Its job is validation + narrative + routing, in that order — not a third sales pitch competing with its own children.

### "I" vs. "we" — keep it, on purpose
`/plm` is written in first person ("I," a named senior consultant, Corp-to-Corp). `/foundry` is written as a company ("we," a product/system with support behind it). This looks like an inconsistency to fix. It isn't — it's correct segmentation, and worth stating explicitly so it doesn't get "harmonized" away by accident later:

- A US/EU engineering director hiring **fractional PLM help** is buying a *de-risked individual* — a named senior person is the whole pitch, versus the black box of a staffing agency.
- An Indian plant manager deploying **a monitoring system on their floor** is buying *a company that will still be there* for support, service, and a second install — a solo operator is the risk here, not the reassurance.

The hub reconciles this by speaking in company voice ("we"/Gompa Technologies) while anchoring both practices to one named founder, so a visitor reads "I" and "we" as the same person in two roles, not two different outfits sharing a domain. The reusable line for this, anywhere the hub needs it:

> *Built and led by Siddharth Solanki — the same engineer who architected PLM for Google's hardware division now runs the practice that watches Indian foundry floors in real time.*

(Name is already public on `/plm` via the Calendly embed title — using it here isn't a new disclosure, just consistency.)

---

## 3. The signature move

Per the instinct to avoid a flattened, one-size-fits-all skin: **don't invent a third visual style that belongs to neither page.** Instead, let the hub's own "Two Practices" panels visually preview their destination — a literal visual pun on *bridge*. The PLM panel is rendered in `/plm`'s cool ink/parchment/schematic/mono language; the Foundry panel is rendered in `/foundry`'s near-black/molten/stencil language. Clicking through then feels continuous instead of jarring, because the panel already looked like where you're headed.

The hub's own chrome (nav, footer, hero, the connective sections that belong to neither practice) sits in a third register that's genuinely a *blend*, not a coin-flip toward one side:

- **Hero graphic concept (the one bold swing on this page):** a single SVG motif that states the thesis wordlessly — `/plm`'s orthogonal schematic node-and-line drawing, whose lines dissolve rightward/downward into `/foundry`'s warm radial glow. Digital thread meets shop floor, in one image, used once. This is the thing this page should be remembered for — keep everything else around it quiet.
- **A fixable gap worth flagging:** `/plm` has a well-made SVG favicon (a compass/target mark in ink + accent); `/foundry` currently ships with none. Give all three pages the same favicon/logo mark in the code pass — cheap fix, real consistency win.

---

## 4. Design token bridge

Both palettes already share an accent-orange family and a dark-mode-first instinct — that's a gift, not a coincidence to smooth over. The master tokens below are proposed starting points (informed by literal midpoints between the two existing palettes), not final production values — tune these by eye once you're in code.

| Token | `/plm` (current) | `/foundry` (current) | **Master (proposed)** |
|---|---|---|---|
| Background, primary | `#0E1420` cool ink | `#0F0D0B` warm near-black | `#101319` graphite |
| Background, alt | `#161E2C` | `#141210` | `#161821` |
| Card surface | `#1D2836` | `#1B1814` | `#1C1F28` |
| Light/paper section | `#EDEAE2` | *(none)* | `#EDEAE2` — retained, used sparingly where the hub needs contrast (e.g. §7 Proof) |
| Accent | `#E28D3D` | `#E07818` | `#E2822E` — "the Zero Deviation orange" |
| Accent, deep | `#9C4A1E` | `#5C2E06` | `#7A3C14` |
| Secondary accent | `#6B8CAE` blue-grey | *(none)* | Retained, but scoped to PLM-flavored components only |
| Display type | Space Grotesk | Big Shoulders Display | **Space Grotesk** — primary; Big Shoulders reserved as an accent face inside Foundry-flavored components only (stat numerals, that panel's headline) |
| Body type | IBM Plex Sans | DM Sans | **IBM Plex Sans** — primary, for consistency |
| Mono / label type | IBM Plex Mono | *(DM Sans for small labels)* | **IBM Plex Mono** — carried brand-wide; it's doing real work signaling "engineering precision" and both audiences read it that way |
| Corner radius | 3–10px | 3px (tight) | 3–8px, leaning tight/industrial for shared chrome |
| Imagery | None — graphic/typographic only | None — graphic/typographic only | **Keep it that way.** Both pages already commit to no stock photography; the hub shouldn't be the page that breaks that discipline. |

---

## 5. Content architecture for gompatech.com

Nine sections. Order is deliberate: fast exits early (for people who already know why they're here), depth in the middle (for people still forming the picture), a second chance to convert at the bottom.

### A. Header / nav
**Job:** fast wayfinding, calm chrome. The fork happens in the hero, not up here — a nav bar juggling two competing CTAs reads as indecisive.

- Logo: Gompa**Tech** wordmark
- Nav tag (small, under/beside logo): *Engineering Systems & Manufacturing Intelligence*
- Links: How We Help · Why Gompa · Proof · Contact
- No CTA button in the nav itself.

### B. Hero
**Job:** land the shared idea before anyone has to self-select, then hand a fast exit to whoever already knows their door.

```
EYEBROW   Two Fronts, One Discipline

H1        Define the Process. Defend the Process.
          Zero Deviation.

SUB       Manufacturing rarely fails at the point everyone's watching.
          It fails in the gap between what a process was designed to do
          and what it's actually doing — inside the engineering systems
          that define how a product gets built, or on the floor where
          it's actually built. Gompa Technologies closes that gap,
          wherever it opens up.

CTA       [ PLM & Teamcenter Consulting → ]   [ Foundry Process Intelligence → ]

STRIP     15+ yrs enterprise PLM (Google · Boeing · T.D. Williamson)
          — 4-month live foundry pilot, Tier-1 OEM supply chain
```

Wireframe (schematic-to-glow hero graphic sits right, per §3):
```
┌──────────────────────────────────────┬─────────────┐
│ EYEBROW                              │   ·  ·      │
│ DEFINE THE PROCESS. DEFEND THE       │  ·      ·   │  ← schematic lines,
│ PROCESS. ZERO DEVIATION.             │   ·    ·    │     warming to a
│ [sub paragraph]                      │    ·  ·     │     glow lower-right
│ [PLM & Teamcenter →] [Foundry →]     │     ··      │
│ [mono credibility strip]             │             │
└──────────────────────────────────────┴─────────────┘
```

### C. The Space Between
**Job:** this is the "prospect relatability" section — prove the thesis with two specific, recognizable scenes side by side, before asking either audience for anything.

```
EYEBROW   The Pattern
H2        You don't lose money at the point of failure.
          You lose it in the silence before anyone notices.

┌─ Inside your engineering systems ──┐  ┌─ On your production floor ─────┐
│ A workflow stalls in Teamcenter    │  │ A step gets missed mid-shift.  │
│ and nobody owns it. A migration    │  │ An addition goes in slightly   │
│ drags because the team's stretched │  │ out of spec. Nobody flags it — │
│ thin. Six months later it's not a  │  │ there was nothing there to     │
│ line item — it's a program delay   │  │ flag it. Weeks later it's not  │
│ everyone's asking about.           │  │ a process note — it's a        │
│                                     │  │ rejection call from your       │
│                                     │  │ biggest customer.              │
└─────────────────────────────────────┘  └─────────────────────────────────┘

CLOSING   Different floor. Same failure mode: a process everyone assumed
          was being followed, and no reliable way to know if it actually
          was — until it already cost you.
```

Design note: this is where the split-styling from §3 can start — left column nudges toward the ink/mono register, right column toward the near-black/stencil register, without fully committing yet (that's the next section).

### D. Two Practices, One Discipline
**Job:** the actual offer. Two problem-first doors, each fully in that audience's own voice — not a capability list.

**Card A — styled in `/plm`'s register**
> **For Engineering, PLM & IT Leaders — US / Europe**
> ### PLM & Teamcenter Advisory
> Your Teamcenter backlog doesn't wait for a req to close.
>
> Fractional support, embedded augmentation, or fixed-scope delivery from a 15-year Teamcenter architect. Corp-to-Corp, delivered remotely, no sponsorship required.
>
> *Zero-downtime migrations · Dispatcher rebuilds · 15+ yrs enterprise PLM*
>
> **See how the practice works →** `/plm`

**Card B — styled in `/foundry`'s register**
> **For Plant & Quality Leaders — India**
> ### Foundry Process Intelligence
> The first sign of a deviation shouldn't be a rejection call.
>
> Computer-vision monitoring that catches SOP deviations on the line, in real time — before the heat is poured, not after the audit.
>
> *4-month live pilot · Tier-1 OEM supply chain · 24/7 automated monitoring*
>
> **See how the system works →** `/foundry`

### E. Why One Roof
**Job:** pre-empt the "wait, why does a Teamcenter guy also sell foundry cameras?" question and turn it into the strongest credibility asset on the page, before it becomes an objection.

```
EYEBROW   Why Both Live Here
H2        The same discipline, applied to both ends of the product's life.

Gompa Technologies is built and led by Siddharth Solanki, a systems
engineer who spent 15+ years architecting PLM and engineering-systems
infrastructure for manufacturers including Google's hardware division,
Boeing, and T.D. Williamson — the kind of work where an ungoverned
workflow costs real money long before anyone notices. Back in India,
that same instinct — find where defined process and actual practice
diverge, then close the gap — went looking for a new expression, and
found one: a computer-vision system that watches the shop floor itself,
catching SOP deviations the moment they happen, not the moment they
cost you a customer.

Two different tools. One diagnostic question behind both: where is
your process actually deviating from what it's supposed to be?

Full background on the PLM practice → /plm#experience
```

### F. Proof
**Job:** back the story with real numbers — as two honest, clearly-labeled clusters, not blended into one list that implies Google and Boeing were clients.

```
CAREER HERITAGE                    |  FOUNDRY PILOT PROOF
15+ Yrs Enterprise PLM Delivery    |  4-Month Live Pilot
Google · Boeing · T.D. Williamson  |  Tier-1 OEM Supply Chain
Zero-Downtime 2007→8.3 Migration   |  0 Dedicated Operators Needed
2× Rendering Throughput            |  24/7 Automated Floor Monitoring
```
**Flag, don't fake:** neither existing page has a named client testimonial (the foundry pilot description is close, but it's a description, not an attributed quote). Don't manufacture one. Structure this section so a real quote slots in easily once you have one — in the meantime the numbers above are real and specific, which does more work than an invented quote would anyway.

### G. The Thread Ahead
**Job:** signal range for AIoT, Continuous Monitoring, Digital Thread, Industry 4.0, and broader Engineering Systems work — the "secondary services" you're at liberty to add — without recreating the capability-dump problem this whole exercise exists to fix.

```
EYEBROW   Where This Is Headed
H2        The same thread, extending.

PLM and shop-floor monitoring are two ends of a single idea: the
digital thread connecting how a product is engineered to how it's
actually built. As that thread extends, so does the work — continuous
equipment monitoring, broader AIoT deployments across plant
operations, Industry 4.0 integration beyond Teamcenter. If your gap
lives somewhere on that thread, it's worth a conversation even if it
doesn't fit either box above.

CHIPS     [ AIoT ]  [ Continuous Monitoring ]  [ Digital Thread ]
          [ Industry 4.0 Integration ]  [ Engineering Systems Consulting ]
```
Deliberately a strip of chips plus one paragraph — **not** five more full tiles. See §7 for why this restraint is load-bearing, not a placeholder to fill in later.

### H. Get in Touch
**Job:** route, don't duplicate the contact flows that already exist and already work (Calendly+email on `/plm`, form+WhatsApp on `/foundry`).

```
EYEBROW   Get in Touch
H2        Tell us which floor you're calling from.

[ PLM / Teamcenter Inquiry → ]        routes to /plm#contact
[ Foundry / Manufacturing Inquiry → ] routes to /foundry#contact

Something else? Write to hello@gompatech.com — every message
reaches Siddharth directly.
```
(`hello@gompatech.com` is a new suggested alias, distinct from `plm@` and `info@`, for whatever doesn't cleanly fit either box.)

### I. Footer
```
GOMPATECH
Define the Process · Defend the Process · Zero Deviation
PLM & Teamcenter (/plm)  ·  Foundry Intelligence (/foundry)  ·  Email
© 2026 Gompa Technologies · Ujjain, Madhya Pradesh, India
```
Exact structural echo of both existing footers — same pattern, master-level tagline.

---

## 6. Naming options worth considering

The computer-vision SOP system currently has no proper name — it's "the system" throughout `/foundry`. Products with names feel ownable and fundable; products described only as "the system" read as a one-off. A few options, in case a name is worth committing to before the next build:

| Name | Rationale | Trade-off |
|---|---|---|
| **Gompa Sentinel** | Vigilance/watching connotation; pairs naturally with the wordmark; portable beyond foundries | Slightly more abstract than a plant manager might want on first read |
| **LineWatch** | Literal, instantly understood by a non-technical plant manager, works fine mixed into Hindi/English shop-floor conversation | Less distinctive as a standalone brand |
| **SentryCast** | Nods to "casting," foundry-specific flavor | Less portable if/when you expand past foundries |
| *(status quo)* "Foundry Intelligence" as descriptor, no product name | Zero risk, zero new decision | Reads as a category, not a product, when the pitch eventually needs to sound fundable/ownable |

Given the brief already frames foundries as the *current* beachhead rather than the ceiling, **Gompa Sentinel** or **LineWatch** travel better than a foundry-specific name. This is optional — the architecture above works fine with the current descriptive naming if you'd rather not decide this now.

---

## 7. Guardrails

Things worth protecting on purpose, since they're each an easy thing to accidentally undo in a later pass:

- **Don't expand §G's chips into full tiles** until at least one of them has a real, specific proof point behind it (a delivered engagement, a pilot, anything as concrete as the foundry pilot stats). That's the exact capability-dump pattern this whole brief exists to undo — the fastest way back to the two-tiles problem is five vague tiles instead of two.
- **Don't fabricate testimonials or quotes.** Real numbers you have beat invented quotes you don't.
- **Don't collapse "I" vs. "we"** into one voice across all three pages — see §2. It's doing real segmentation work, not sloppiness.
- **Don't introduce stock photography.** Both existing pages commit fully to graphic/typographic visual language; the hub is the wrong place to break that discipline first.
- **Resist generic dark-mode-SaaS defaults** in the code pass (near-black + a single acid accent with no other point of view, or a warm-cream-and-serif template look) — the graphite/orange bridge above is derived from your own two pages, which is exactly what makes it distinctive rather than templated.
- **Keep the hub's pillar copy problem-first.** The urge to list every tag and competency at the hub level should be resisted — that's what `/plm`'s Expertise section and `/foundry`'s How It Works section are already for. The hub should never try to out-detail its own children.

---

## 8. SEO / meta (for whenever this gets built)

```html
<title>Gompa Technologies — Engineering Systems & Manufacturing Process Intelligence</title>
<meta name="description" content="PLM & Teamcenter consulting for US/Europe engineering leaders, and real-time computer-vision process intelligence for Indian manufacturers. One discipline: closing the gap between defined process and actual execution.">
```

---

## 9. Open decisions for you

Everything above is a decisive point of view, not a menu — but three things are genuine judgment calls where I've made an assumption on your behalf:

1. **Product naming (§6)** — commit to a name now, or keep it descriptive until there's more to build around?
2. **Using "Siddharth Solanki" by name on the hub's "Why One Roof" section** — I've assumed yes, since it's already public on `/plm` and it's the credibility anchor that makes the "I"/"we" duality legible. Say if you'd rather keep the hub more institutional.
3. **Greenlighting §G (The Thread Ahead) now** vs. holding it back entirely until one adjacent line has real delivery proof — both are defensible; I lean toward including it lightly now (as written, chips-only) rather than waiting.

## 10. Next steps

This doc is the brief. When you're ready to build the actual page, the two flagship pages already tell you almost everything about implementation patterns (scroll-reveal, mobile-first for `/foundry`-flavored components, the titleblock/schematic device from `/plm`) — the hub mostly needs to borrow both intelligently rather than invent a third system from scratch.
