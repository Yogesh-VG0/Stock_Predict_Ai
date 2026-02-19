import type { Metadata } from "next"
import React from "react"
import Link from "next/link"
import AnimatedBlock from "@/components/marketing/animated-block"
import MarketingShell from "@/components/marketing/marketing-shell"

export const metadata: Metadata = {
  title: "Disclaimer | StockPredict AI",
  description:
    "Educational disclaimer for StockPredict AI. Not financial advice. Predictions are probabilistic estimates and may be inaccurate.",
}

export default function DisclaimerPage() {
  return (
    <MarketingShell activeNav="disclaimer">
      <div className="space-y-10">
        <AnimatedBlock>
          <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/70 to-black p-6 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-950/50 px-3 py-1 text-xs text-zinc-400 mb-4">
              Legal / educational
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">Disclaimer</h1>
            <p className="mt-3 text-sm sm:text-base text-zinc-400 max-w-3xl leading-relaxed">
              This page explains what StockPredict AI is (and is not).
            </p>

            <div className="mt-6 flex flex-col sm:flex-row gap-3">
              <Link
                href="/how-it-works"
                className="inline-flex items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900/60 px-4 py-2.5 text-sm text-white transition-colors"
              >
                How it works
              </Link>
              <Link
                href="/methodology"
                className="inline-flex items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900/60 px-4 py-2.5 text-sm text-white transition-colors"
              >
                Methodology
              </Link>
            </div>
          </div>
        </AnimatedBlock>

        <AnimatedBlock delay={0.06}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Educational Purposes Only")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The content provided on this website, including but not limited to stock data, analysis, and stock picks, is for informational and educational purposes only. The website does not provide any form of investment, financial, or trading advice. The information is not intended to be used for the purpose of making or refraining from making any investment decisions.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.08}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "No Recommendations or Endorsements")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The content on this site does not constitute a recommendation or endorsement to buy or sell any particular security or investment. The website does not make any representation or guarantee regarding the accuracy, timeliness, or completeness of any stock data or information provided. You should not rely solely on any information on this website for making any trading or investment decisions.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.1}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Stock Price Accuracy")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Stock prices and other financial data displayed on this website are for informational purposes only. These prices may be delayed, inaccurate, or subject to change without notice. Always verify stock prices and other market data with a reliable source before making any decisions.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.12}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Risks of Trading and Investments")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Trading stocks, options, and other financial instruments involves a high level of risk and may not be suitable for all investors. There is no guarantee of future performance, and you could lose some or all of your invested capital. Past performance is not necessarily indicative of future results.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.14}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Responsibility for Decision-Making")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              By using this website, you acknowledge and agree that you are solely responsible for your investment decisions. We strongly recommend that you conduct your own research and seek the advice of a qualified financial advisor before making any investment, trading, or financial decisions.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.16}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Limitation of Liability")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              In no event shall the website, its owners, employees, affiliates, or partners be liable for any direct, indirect, incidental, special, consequential, or any other damages arising out of your use or reliance on this website. This includes, without limitation, any losses or damages related to trading or investment decisions made based on the content of this site.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.18}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "No Client Relationship")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Use of this website does not create any client, advisory, fiduciary, or professional relationship between you and the website owner. We are not registered investment advisors, brokers, or dealers.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.2}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Third-Party Links")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              This website may contain links to third-party websites. We do not control, endorse, or take responsibility for the content, services, or practices of any third-party site. Accessing these links is at your own risk.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.22}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Changes to Content")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The content provided on this website is subject to change at any time without notice. We do not guarantee that the information is always current, complete, or error-free.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.24}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "No Guarantees")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              There are no guarantees made regarding the outcome of any trading or investment strategies discussed or provided on this site. All decisions made based on the information provided on this website are made at your own risk.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.26}>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-5 sm:p-6">
            <div className="text-sm text-zinc-400">
              Want details on the pipeline? Read{" "}
              <Link href="/how-it-works" className="text-emerald-400 hover:text-emerald-300 transition-colors">
                How it works
              </Link>{" "}
              and{" "}
              <Link href="/methodology" className="text-emerald-400 hover:text-emerald-300 transition-colors">
                Methodology
              </Link>
              .
            </div>
          </div>
        </AnimatedBlock>
      </div>
    </MarketingShell>
  )
}

