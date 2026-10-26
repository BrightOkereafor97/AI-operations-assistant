import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import CognitoAuthProvider from "./auth-provider";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Operations Intelligence",
  description: "AI Operations Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <CognitoAuthProvider>
          {children}
        </CognitoAuthProvider>
      </body>
    </html>
  );
}