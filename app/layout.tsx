import type { Metadata } from "next"

export const metadata: Metadata = { title: "Form/90 — Premier League Predictor", description: "A transparent statistical view of every Premier League match." }

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en" className="bg-[#0b0b10]"><body>{children}</body></html> }
