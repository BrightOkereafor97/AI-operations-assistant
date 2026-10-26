"use client";

import Link from "next/link";
import { useAuth } from "react-oidc-context";

export default function AuthCallbackPage() {
    const auth = useAuth();

    return (
        <main className="flex min-h-screen items-center justify-center bg-slate-950 px-4 text-white">
            <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <h1 className="text-xl font-semibold">
                    {auth.error
                        ? "Sign-in could not be completed"
                        : "Completing sign-in"}
                </h1>

                <p className="mt-3 break-words text-sm text-slate-300">
                    {auth.error
                        ? auth.error.message
                        : "Please wait while we complete your sign-in."}
                </p>

                {auth.error && (
                    <Link
                        href="/"
                        className="mt-6 inline-block rounded-xl bg-blue-600 px-4 py-3 text-sm font-medium hover:bg-blue-500"
                    >
                        Return to home
                    </Link>
                )}
            </div>
        </main>
    );
}