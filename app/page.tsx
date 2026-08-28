"use client"

import { useEffect, useState } from "react"
import { ArrowUpRight, Check, ChevronRight, CircleHelp, Menu, X, XCircle } from "lucide-react"
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import "./styles.css"

// Set this to wherever your FastAPI server is running.
// Local dev default matches: uv run uvicorn api.server:app --reload --port 8000
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

type Match = { match_id:string; home_team:{name:string; crest_url:string}; away_team:{name:string; crest_url:string}; kickoff_at:string; home_win_probability:number; draw_probability:number; away_win_probability:number; predicted_home_goals:number; predicted_away_goals:number; most_likely_home_goals:number; most_likely_away_goals:number; scoreline_probabilities:Record<string,number>; explanation_summary:string|null }

// Raw shape returned by GET /fixtures/current-gameweek — field names differ
// slightly from the frontend's Match type (expected_* vs predicted_*, and
// no kickoff_at yet since that's not in the feature CSV).
type ApiFixture = {
  match_id: string
  home_team: { name: string; short_name: string | null; crest_url: string | null }
  away_team: { name: string; short_name: string | null; crest_url: string | null }
  expected_home_goals: number
  expected_away_goals: number
  home_win_probability: number
  draw_probability: number
  away_win_probability: number
  most_likely_home_goals: number
  most_likely_away_goals: number
  scoreline_probabilities: Record<string, number>
  explanation_summary: string | null
  gameweek: number
  season: number
}

type ApiHistoryMatch = {
  match_id: string
  home_team: { name: string; short_name: string | null; crest_url: string | null }
  away_team: { name: string; short_name: string | null; crest_url: string | null }
  actual_home_goals: number
  actual_away_goals: number
  predicted_home_goals: number
  predicted_away_goals: number
  home_win_probability: number
  draw_probability: number
  away_win_probability: number
  predicted_outcome: "HOME" | "DRAW" | "AWAY"
  actual_outcome: "HOME" | "DRAW" | "AWAY"
  outcome_correct: boolean
  scoreline_correct: boolean
}

type ApiHistorySummary = {
  total_matches: number
  outcome_correct: number
  scoreline_correct: number
  outcome_accuracy: number | null
  scoreline_accuracy: number | null
}

const crest=(initials:string, color:string)=>`data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80"><path fill="${color}" d="M8 8h64v40c0 17-14 26-32 30C22 74 8 65 8 48V8Z"/><text x="40" y="48" text-anchor="middle" fill="white" font-family="Arial" font-size="20" font-weight="700">${initials}</text></svg>`)}`

// Deterministic fallback initials/color so a match still looks fine if
// crest_url comes back null (e.g. Supabase not configured locally).
function fallbackCrest(name: string): string {
  const initials = name.replace(/FC|AFC|United|City/g, "").trim().split(" ").map(w => w[0]).join("").slice(0, 3).toUpperCase() || "?"
  let hash = 0
  for (const ch of name) hash = ch.charCodeAt(0) + ((hash << 5) - hash)
  const color = `hsl(${hash % 360}, 55%, 40%)`
  return crest(initials, color)
}

function teamDisplay(t: { name: string; short_name: string | null; crest_url: string | null }) {
  return { name: t.short_name || t.name, crest_url: t.crest_url || fallbackCrest(t.name) }
}

function mapFixture(f: ApiFixture): Match {
  return {
    match_id: f.match_id,
    home_team: teamDisplay(f.home_team),
    away_team: teamDisplay(f.away_team),
    // kickoff_at isn't in the feature data yet — omit gracefully.
    kickoff_at: "",
    home_win_probability: f.home_win_probability,
    draw_probability: f.draw_probability,
    away_win_probability: f.away_win_probability,
    predicted_home_goals: f.expected_home_goals,
    predicted_away_goals: f.expected_away_goals,
    most_likely_home_goals: f.most_likely_home_goals,
    most_likely_away_goals: f.most_likely_away_goals,
    scoreline_probabilities: f.scoreline_probabilities,
    explanation_summary: f.explanation_summary,
  }
}

const pct=(n:number)=>`${Math.round(n*100)}%`
const time=(iso:string)=>iso ? new Date(iso).toLocaleDateString("en-GB",{weekday:"short",day:"numeric",month:"short"}) : ""

function FixtureCard({ m, index, total, onSelect }: { m: Match; index: number; total: number; onSelect: (m: Match) => void }) {
  return (
    <section className="featured" key={m.match_id}>
      <div className="section-label">FIXTURE <span>{String(index + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}</span></div>
      <div className="feature-grid">
        <div className="fixture-title">
          <div className="fixture-meta">{time(m.kickoff_at) || "THIS GAMEWEEK"}</div>
          <h2>{m.home_team.name}<br /><span>vs</span> {m.away_team.name}</h2>
          <p>{m.explanation_summary}</p>
          <button className="outline" onClick={() => onSelect(m)}>View full prediction <ArrowUpRight /></button>
        </div>
        <div className="probability">
          <div className="crest-row">
            <img src={m.home_team.crest_url} alt={`${m.home_team.name} crest`} />
            <div className="versus">VS</div>
            <img src={m.away_team.crest_url} alt={`${m.away_team.name} crest`} />
          </div>
          <div className="big-prob"><strong>{pct(Math.max(m.home_win_probability, m.draw_probability, m.away_win_probability))}</strong><span>{m.home_win_probability >= m.away_win_probability && m.home_win_probability >= m.draw_probability ? "home win" : m.away_win_probability >= m.draw_probability ? "away win" : "draw"}</span></div>
          <div className="bar">
            <i style={{ width: `${m.home_win_probability * 100}%` }} />
            <i style={{ width: `${m.draw_probability * 100}%` }} />
            <i style={{ width: `${m.away_win_probability * 100}%` }} />
          </div>
          <div className="bar-labels">
            <span>{m.home_team.name.toUpperCase()} {pct(m.home_win_probability)}</span>
            <span>DRAW {pct(m.draw_probability)}</span>
            <span>{m.away_team.name.toUpperCase()} {pct(m.away_win_probability)}</span>
          </div>
        </div>
      </div>
    </section>
  )
}

function OverviewPage({ matches, gameweek, onSelect }: { matches: Match[]; gameweek: number | null; onSelect: (m: Match) => void }) {
  return (
    <>
      <section className="hero" id="overview">
        <div className="eyebrow"><span className="live-dot" /> GAMEWEEK {gameweek !== null ? String(gameweek).padStart(2, "0") : "—"} <span className="slash">/</span> {new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }).toUpperCase()}</div>
        <h1>Every match.<br /><em>Read differently.</em></h1>
        <p className="lede">A transparent statistical view of the Premier League — powered by a Poisson/Dixon-Coles model.</p>
      </section>
      {matches.map((m, i) => <FixtureCard key={m.match_id} m={m} index={i} total={matches.length} onSelect={onSelect} />)}
    </>
  )
}

// --------------------------------------------------------------------------
// History > Performance (existing chart, wired to /accuracy)
// --------------------------------------------------------------------------

// Accuracy-over-time chart data has no real backing table yet
// (accuracy_snapshots is empty in Supabase) — kept as illustrative
// placeholder until that table is populated. The three headline metric
// numbers below the chart ARE live from /accuracy.
const accuracySeries=[{week:"W1",model:38,bookmaker:41},{week:"W2",model:42,bookmaker:44},{week:"W3",model:40,bookmaker:43},{week:"W4",model:46,bookmaker:45},{week:"W5",model:44,bookmaker:46},{week:"W6",model:48,bookmaker:47},{week:"W7",model:47,bookmaker:48},{week:"W8",model:50,bookmaker:48}]

function PerformancePanel({ modelAccuracy, logLoss, matchesEvaluated }: { modelAccuracy: number | null; logLoss: number | null; matchesEvaluated: number | null }) {
  return (
    <section className="performance" id="performance">
      <div className="performance-copy">
        <div className="eyebrow">MODEL PERFORMANCE</div>
        <h2>Gets sharper<br /><em>with every week.</em></h2>
        <p>We backtest our predictions against actual results and bookmaker consensus. No black box — just a model that earns its edge.</p>
      </div>
      <div className="chart-panel">
        <div className="chart-top">
          <span>ACCURACY OVER TIME</span>
          <div><b><i className="dot model" /> Model</b><b><i className="dot book" /> Bookmaker</b></div>
        </div>
        <div className="chart">
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={accuracySeries} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
              <XAxis dataKey="week" stroke="var(--muted)" tickLine={false} axisLine={false} />
              <YAxis domain={[30, 55]} stroke="var(--muted)" tickLine={false} axisLine={false} tickFormatter={v => `${v}%`} />
              <Tooltip contentStyle={{ background: "#171722", border: "1px solid #343342", borderRadius: 8 }} formatter={(v) => `${v}%`} />
              <Line type="monotone" dataKey="model" stroke="var(--accent)" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="bookmaker" stroke="var(--muted)" strokeWidth={2} strokeDasharray="4 4" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="metrics">
          <div><strong>{modelAccuracy !== null ? pct(modelAccuracy) : "—"}</strong><span>Model accuracy</span></div>
          <div><strong>{logLoss !== null ? logLoss.toFixed(2) : "—"}</strong><span>Log loss</span></div>
          <div><strong>{matchesEvaluated !== null ? matchesEvaluated : "—"}</strong><span>Matches evaluated</span></div>
        </div>
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------
// History > Past Predictions (new — wired to /history/gameweeks + /history/gameweek/{n})
// --------------------------------------------------------------------------

function PastPredictionsPanel() {
  const [availableGameweeks, setAvailableGameweeks] = useState<number[]>([])
  const [selectedGw, setSelectedGw] = useState<number | null>(null)
  const [gwMatches, setGwMatches] = useState<ApiHistoryMatch[]>([])
  const [gwSummary, setGwSummary] = useState<ApiHistorySummary | null>(null)
  const [loadingList, setLoadingList] = useState(true)
  const [loadingDetail, setLoadingDetail] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function loadList() {
      setLoadingList(true)
      try {
        const res = await fetch(`${API_BASE}/history/gameweeks`)
        if (!res.ok) throw new Error()
        const data = await res.json()
        if (cancelled) return
        const gws: number[] = data.gameweeks ?? []
        setAvailableGameweeks(gws)
        if (gws.length > 0) setSelectedGw(gws[gws.length - 1])
      } catch {
        if (!cancelled) setAvailableGameweeks([])
      } finally {
        if (!cancelled) setLoadingList(false)
      }
    }

    loadList()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (selectedGw === null) return
    let cancelled = false

    async function loadDetail() {
      setLoadingDetail(true)
      try {
        const res = await fetch(`${API_BASE}/history/gameweek/${selectedGw}`)
        if (!res.ok) throw new Error()
        const data = await res.json()
        if (cancelled) return
        setGwMatches(data.matches ?? [])
        setGwSummary(data.summary ?? null)
      } catch {
        if (!cancelled) { setGwMatches([]); setGwSummary(null) }
      } finally {
        if (!cancelled) setLoadingDetail(false)
      }
    }

    loadDetail()
    return () => { cancelled = true }
  }, [selectedGw])

  if (loadingList) {
    return <section className="history-panel"><p className="history-empty">Checking for completed gameweeks…</p></section>
  }

  if (availableGameweeks.length === 0) {
    return (
      <section className="history-panel">
        <div className="eyebrow">PAST PREDICTIONS</div>
        <p className="history-empty">
          No gameweeks are ready to review yet. Once a gameweek finishes and results are pulled in,
          it&apos;ll show up here with a fair, no-hindsight backtest — the model is retrained excluding
          that gameweek before predicting it, so it&apos;s never seen the answer in advance.
        </p>
      </section>
    )
  }

  return (
    <section className="history-panel">
      <div className="section-heading">
        <div>
          <div className="eyebrow">PAST PREDICTIONS</div>
          <h2>How we did</h2>
        </div>
        <div className="gw-switcher">
          {availableGameweeks.map(gw => (
            <button key={gw} className={selectedGw === gw ? "active" : ""} onClick={() => setSelectedGw(gw)}>
              GW{gw}
            </button>
          ))}
        </div>
      </div>

      {gwSummary && (
        <div className="metrics history-metrics">
          <div><strong>{gwSummary.outcome_accuracy !== null ? pct(gwSummary.outcome_accuracy) : "—"}</strong><span>Outcome accuracy</span></div>
          <div><strong>{gwSummary.scoreline_accuracy !== null ? pct(gwSummary.scoreline_accuracy) : "—"}</strong><span>Exact scoreline</span></div>
          <div><strong>{gwSummary.total_matches}</strong><span>Matches evaluated</span></div>
        </div>
      )}

      {loadingDetail ? (
        <p className="history-empty">Loading gameweek {selectedGw}…</p>
      ) : (
        <div className="history-list">
          {gwMatches.map(m => {
            const home = teamDisplay(m.home_team)
            const away = teamDisplay(m.away_team)
            return (
              <div className="history-row" key={m.match_id}>
                <div className="history-teams">
                  <img src={home.crest_url} alt="" />
                  <span>{home.name}</span>
                  <strong>{m.actual_home_goals}–{m.actual_away_goals}</strong>
                  <span>{away.name}</span>
                  <img src={away.crest_url} alt="" />
                </div>
                <div className="history-predicted">
                  Predicted {m.predicted_home_goals}–{m.predicted_away_goals}
                </div>
                <div className="history-badges">
                  <span className={`badge ${m.outcome_correct ? "badge-good" : "badge-bad"}`}>
                    {m.outcome_correct ? <Check size={14} /> : <XCircle size={14} />} Outcome
                  </span>
                  <span className={`badge ${m.scoreline_correct ? "badge-good" : "badge-bad"}`}>
                    {m.scoreline_correct ? <Check size={14} /> : <XCircle size={14} />} Scoreline
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

function HistoryPage({ modelAccuracy, logLoss, matchesEvaluated }: { modelAccuracy: number | null; logLoss: number | null; matchesEvaluated: number | null }) {
  const [subTab, setSubTab] = useState<"Performance" | "Past Predictions">("Performance")

  return (
    <>
      <section className="hero hero-compact" id="history">
        <div className="eyebrow">TRACK RECORD</div>
        <h1>How the model<br /><em>holds up.</em></h1>
        <div className="sub-tabs">
          {(["Performance", "Past Predictions"] as const).map(t => (
            <button key={t} className={subTab === t ? "active" : ""} onClick={() => setSubTab(t)}>{t}</button>
          ))}
        </div>
      </section>
      {subTab === "Performance"
        ? <PerformancePanel modelAccuracy={modelAccuracy} logLoss={logLoss} matchesEvaluated={matchesEvaluated} />
        : <PastPredictionsPanel />
      }
    </>
  )
}

export default function Page(){
 const [tab,setTab]=useState<"Overview"|"History">("Overview")
 const [selected,setSelected]=useState<Match|null>(null)
 const [mobile,setMobile]=useState(false)

 const [matches,setMatches]=useState<Match[]>([])
 const [gameweek,setGameweek]=useState<number|null>(null)
 const [loading,setLoading]=useState(true)
 const [error,setError]=useState<string|null>(null)

 const [modelAccuracy,setModelAccuracy]=useState<number|null>(null)
 const [logLoss,setLogLoss]=useState<number|null>(null)
 const [matchesEvaluated,setMatchesEvaluated]=useState<number|null>(null)

 useEffect(()=>{
  let cancelled=false

  async function load(){
   setLoading(true)
   setError(null)
   try{
    const res=await fetch(`${API_BASE}/fixtures/current-gameweek`)
    if(!res.ok) throw new Error(`API returned ${res.status}`)
    const data=await res.json()
    if(cancelled) return
    const mapped:Match[]=(data.fixtures as ApiFixture[]).map(mapFixture)
    setMatches(mapped)
    setGameweek(data.gameweek ?? null)
   }catch(err){
    if(!cancelled) setError(err instanceof Error ? err.message : "Failed to load fixtures")
   }finally{
    if(!cancelled) setLoading(false)
   }
  }

  async function loadAccuracy(){
   try{
    const res=await fetch(`${API_BASE}/accuracy`)
    if(!res.ok) return
    const data=await res.json()
    if(cancelled) return
    setModelAccuracy(data.model_accuracy ?? null)
    setLogLoss(data.model_log_loss ?? null)
    setMatchesEvaluated(data.matches_evaluated ?? null)
   }catch{
    // Non-critical — headline metrics just stay blank if this fails.
   }
  }

  load()
  loadAccuracy()

  return ()=>{cancelled=true}
 },[])

 const nav=(x:"Overview"|"History")=>{setTab(x); setMobile(false); window.scrollTo({top:0,behavior:"smooth"})}

 if(loading){
  return <main className="state-screen"><p>Loading this gameweek&apos;s predictions…</p></main>
 }

 if(error || matches.length===0){
  return <main className="state-screen">
   <p>Couldn&apos;t reach the prediction API{error ? `: ${error}` : "."}</p>
   <p className="state-hint">Make sure the FastAPI server is running at {API_BASE} (uv run uvicorn api.server:app --reload --port 8000).</p>
  </main>
 }

 return <main>
 <header>
  <div className="brand"><span className="brand-mark">P</span><span>FORM<span className="accent">/</span>90</span></div>
  <nav className={mobile?"open":""}>{(["Overview","History"] as const).map(x=><button key={x} className={tab===x?"active":""} onClick={()=>nav(x)}>{x}</button>)}</nav>
  <button className="menu" aria-label="Toggle menu" onClick={()=>setMobile(!mobile)}>{mobile?<X/>:<Menu/>}</button>
  <button className="season">2026/27 <ChevronRight/></button>
 </header>

 {tab==="Overview"
   ? <OverviewPage matches={matches} gameweek={gameweek} onSelect={setSelected} />
   : <HistoryPage modelAccuracy={modelAccuracy} logLoss={logLoss} matchesEvaluated={matchesEvaluated} />
 }

 <footer><div className="brand"><span className="brand-mark">P</span><span>FORM<span className="accent">/</span>90</span></div><span>Built for the curious fan.</span><span>Data refreshes every gameweek <CircleHelp/></span></footer>

 {selected&&<div className="overlay" onClick={()=>setSelected(null)}><article className="modal" onClick={e=>e.stopPropagation()}><button className="close" onClick={()=>setSelected(null)} aria-label="Close"><X/></button><div className="eyebrow">MATCH PREDICTION {time(selected.kickoff_at) && <span>{time(selected.kickoff_at)}</span>}</div><h2>{selected.home_team.name} <em>vs</em> {selected.away_team.name}</h2><div className="modal-score"><div><img src={selected.home_team.crest_url} alt=""/><strong>{selected.most_likely_home_goals}</strong></div><span>—</span><div><strong>{selected.most_likely_away_goals}</strong><img src={selected.away_team.crest_url} alt=""/></div></div><div className="modal-stats"><div><span>WIN PROBABILITY</span><b>{pct(selected.home_win_probability)} <small>—</small> {pct(selected.draw_probability)} <small>—</small> {pct(selected.away_win_probability)}</b><p>Home <i/> Draw <i/> Away</p></div><div><span>EXPECTED GOALS</span><b>{selected.predicted_home_goals.toFixed(1)} <small>—</small> {selected.predicted_away_goals.toFixed(1)}</b></div></div><h3>Most likely scorelines</h3><div className="score-grid">{Object.entries(selected.scoreline_probabilities).map(([score,p])=><div key={score}><b>{score}</b><span>{pct(p)}</span></div>)}</div><p className="explanation">{selected.explanation_summary}</p></article></div>}
 </main>
}
