import React, { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './index.css'

const comparisonData = [
  {
    name: 'Claude Opus 4.8',
    'Public Benchmark': 89.66,
    'Clinical Domain': 44.00,
  },
  {
    name: 'GLM 5.3',
    'Public Benchmark': 92.86,
    'Clinical Domain': 48.98,
  },
  {
    name: 'GPT 5.6 Sol',
    'Public Benchmark': 89.29,
    'Clinical Domain': 56.10,
  },
  {
    name: 'DeepSeek V4',
    'Public Benchmark': 75.00,
    'Clinical Domain': 53.06,
  },
  {
    name: 'Phi-3.5 (PEFT)',
    'Public Benchmark': null,
    'Clinical Domain': 23.33,
  }
];

const publicCohortMetrics = [
  {
    model: 'GLM 5.3',
    accuracy: '92.86%',
    hallucination: '60.71%',
    ece: '0.6190',
    safety: 3,
  },
  {
    model: 'Claude Opus 4.8',
    accuracy: '89.66%',
    hallucination: '58.62%',
    ece: '0.1264',
    safety: 3,
  },
  {
    model: 'GPT 5.6 Sol',
    accuracy: '89.29%',
    hallucination: '46.43%',
    ece: '0.5833',
    safety: 2,
  },
  {
    model: 'DeepSeek V4',
    accuracy: '75.00%',
    hallucination: '28.57%',
    ece: '0.2016',
    safety: 1,
  },
];

const domainCohortMetrics = [
  {
    model: 'GPT 5.6 Sol',
    accuracy: '56.10%',
    hallucination: '26.83%',
    ece: '0.5610',
    safety: 2,
  },
  {
    model: 'DeepSeek V4',
    accuracy: '53.06%',
    hallucination: '12.24%',
    ece: '0.2357',
    safety: 3,
  },
  {
    model: 'GLM 5.3',
    accuracy: '48.98%',
    hallucination: '30.61%',
    ece: '0.2726',
    safety: 8,
  },
  {
    model: 'Claude Opus 4.8',
    accuracy: '44.00%',
    hallucination: '20.00%',
    ece: '0.4980',
    safety: 7,
  },
  {
    model: 'Phi-3.5 (SCD-PEFT)',
    accuracy: '23.33%',
    hallucination: '8.33%',
    ece: '0.0345',
    safety: 5,
  },
];

function App() {
  const [activeCohort, setActiveCohort] = useState('public')

  return (
    <div className="app-root">
      <header className="site-header">
        <div className="header-inner container">
          <a href="#" className="logo mono">MEDEVAL</a>
          <nav className="nav-links">
            <a href="#methodology">Methodology</a>
            <a href="#benchmark">Benchmarks</a>
            <a href="#safety">Safety Audit</a>
            <a href="https://github.com/TeslaInch/medeval-framework" target="_blank" rel="noreferrer">GitHub</a>
          </nav>
        </div>
      </header>

      <div className="container">
        <main>
          <section className="hero">
          <span className="pill">The Clinical Reliability Audit</span>
          <h1>Evaluating AI<br/>Beyond Accuracy</h1>
          <p>
            MedEval is a multi-dimensional diagnostic framework for exposing large language model failure modes in medicine. 
            We quantify calibration error, contextual hallucinations, and lethal contraindications to measure genuine clinical trustworthiness.
          </p>
          <div style={{ marginTop: '2rem' }}>
            <p className="mono" style={{ fontSize: '1rem', color: 'var(--color-text-main)', padding: '1rem', border: '1px solid var(--color-border)', display: 'inline-block', backgroundColor: '#fafafa' }}>
              $ pip install medeval-framework
            </p>
          </div>
        </section>

        <section id="methodology" className="content-section">
          <h2 className="section-title">The MedEval Methodology</h2>
          <p className="section-text">
            High benchmark accuracy on medical examinations (e.g., MedQA) fails to guarantee clinical reliability. Standard multiple-choice evaluations cannot detect when a model is dangerously overconfident, fabricating plausible falsehoods, or issuing lethal contraindications. MedEval operates as a sequential diagnostic pipeline across four dimensions:
          </p>
          
          <div className="methodology-grid">
            <div className="method-card">
              <h3>Dual-Tier Clinical Safety Engine</h3>
              <p>
                Combines deterministic pattern recognition for explicit contraindications (e.g., vasoconstrictors during a vaso-occlusive crisis) with semantic hazard entailment to catch subtle clinical violations where rigid keywords are absent.
              </p>
            </div>
            <div className="method-card">
              <h3>PubMedBERT Hallucination Audit</h3>
              <p>
                A biomedical cross-encoder (PubMedBERT) verifies generated clinical explanations against medical literature. A conditional label-override penalizes fluent answers when factual contradictions are detected.
              </p>
            </div>
            <div className="method-card">
              <h3>Uncertainty & Calibration</h3>
              <p>
                Calculates Expected Calibration Error (ECE) and Brier scores by aggregating token log-probabilities and perturbation self-consistency to expose models that are "confidently wrong."
              </p>
            </div>
            <div className="method-card">
              <h3>Epistemic & Domain Profiling</h3>
              <p>
                Benchmarks general-purpose foundation models alongside parameter-efficient fine-tuned (PEFT) adapters to assess how domain adaptation impacts calibration safety, hallucination suppression, and epistemic honesty.
              </p>
            </div>
          </div>
        </section>

        <section id="benchmark" className="content-section">
          <h2 className="section-title">Surface Accuracy vs. Clinical Reliability</h2>
          <p className="section-text">
            Frontier LLMs achieve 75%–93% accuracy on standard licensing benchmarks, but surface scores fail to capture domain reliability. When evaluated on specialized clinical disease pathology (Sickle Cell Disease), general models suffer noticeable drops in precision while exhibiting high rates of hallucination and miscalibration.
          </p>
          
          <div className="chart-container">
            <div style={{ width: '100%', height: 450 }}>
              <ResponsiveContainer>
                <BarChart data={comparisonData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eaeaea" />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 13, fill: '#555' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 13, fill: '#555' }} unit="%" domain={[0, 100]} />
                  <Tooltip 
                    cursor={{fill: '#f4f4f5'}}
                    contentStyle={{ borderRadius: '0px', border: '1px solid #eaeaea', boxShadow: 'none', fontSize: '14px' }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '14px', paddingTop: '20px' }}/>
                  <Bar dataKey="Public Benchmark" fill="var(--color-bar-public)" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="Clinical Domain" fill="var(--color-bar-scd)" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mono" style={{ textAlign: 'center', marginTop: '2rem', color: 'var(--color-text-muted)', fontSize: '0.9rem' }}>
              Fig 1. Accuracy Comparison: Public Licensing Cohort (MedQA/PubMedQA) vs. Specialized Disease Cohort (Sickle Cell)
            </p>
          </div>
        </section>

        <section id="safety" className="content-section" style={{ borderBottom: 'none' }}>
          <h2 className="section-title">Safety, Hallucination & Calibration Audit</h2>
          <p className="section-text">
            The fundamental risk in clinical AI is unearned certainty. Even on public licensing questions with high accuracy, frontier models hallucinate facts up to 60.7% of the time with severe calibration errors. In specialized clinical domains, our PEFT adapter prioritizes safety—achieving an ECE of 0.0345 and suppressing hallucinations to 8.33%.
          </p>

          <div style={{ display: 'flex', gap: '1rem', marginBottom: '2.5rem' }}>
            <button 
              type="button"
              onClick={() => setActiveCohort('public')}
              className="pill"
              style={{ 
                cursor: 'pointer',
                marginBottom: 0,
                backgroundColor: activeCohort === 'public' ? 'var(--color-text-main)' : 'transparent',
                color: activeCohort === 'public' ? '#ffffff' : 'var(--color-text-main)',
                transition: 'all 0.2s ease',
                outline: 'none'
              }}
            >
              Public Examination Cohort (MedQA / PubMedQA)
            </button>
            <button 
              type="button"
              onClick={() => setActiveCohort('domain')}
              className="pill"
              style={{ 
                cursor: 'pointer',
                marginBottom: 0,
                backgroundColor: activeCohort === 'domain' ? 'var(--color-text-main)' : 'transparent',
                color: activeCohort === 'domain' ? '#ffffff' : 'var(--color-text-main)',
                transition: 'all 0.2s ease',
                outline: 'none'
              }}
            >
              Clinical Pathology Cohort (Sickle Cell Disease)
            </button>
          </div>

          <table className="data-table">
            <thead>
              <tr>
                <th>Evaluated Model</th>
                <th>Accuracy</th>
                <th>Hallucination Rate</th>
                <th>Expected Calibration Err (ECE)</th>
                <th>Safety Violations</th>
              </tr>
            </thead>
            <tbody>
              {(activeCohort === 'public' ? publicCohortMetrics : domainCohortMetrics).map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td className="mono">{row.accuracy}</td>
                  <td className="mono">{row.hallucination}</td>
                  <td className="mono">{row.ece}</td>
                  <td className="mono">{row.safety}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="mono" style={{ color: 'var(--color-text-muted)', fontSize: '0.88rem', marginTop: '1.5rem', lineHeight: '1.6' }}>
            {activeCohort === 'public' 
              ? 'Key Finding: High exam accuracy masks severe overconfidence. GLM 5.3 (92.86% acc) and GPT 5.6 (89.29% acc) exhibit 60.71% and 46.43% hallucination rates alongside severe ECE (>0.58).'
              : 'Key Finding: The domain-adapted Phi-3.5 PEFT adapter drastically suppresses hallucinations (8.33%) and achieves near-perfect calibration (0.0345 ECE), exhibiting epistemic honesty rather than hazardous guessing.'}
          </p>
        </section>
      </main>

      <footer className="footer mono">
        MedEval Framework © 2026. Built for Rigorous Clinical AI Evaluation.
      </footer>
      </div>
    </div>
  )
}

export default App
