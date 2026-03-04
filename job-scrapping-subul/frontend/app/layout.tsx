import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Career Assistant",
  description: "AI-powered career coach",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
