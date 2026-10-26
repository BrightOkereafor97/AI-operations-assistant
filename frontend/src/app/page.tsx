"use client";

import { useState } from "react";
import { useAuth } from "react-oidc-context";
import Dashboard from "./dashboard";

export default function Home() {
    const auth = useAuth();
    const [signInError, setSignInError] = useState<string | null>(null);

    async function signIn() {
        setSignInError(null);

        try {
            await auth.signinRedirect();
        } catch (error) {
            setSignInError(
                error instanceof Error
                    ? error.message
                    : "Unable to open the sign-in page."
            );
        }
    }

    if (auth.isLoading || auth.activeNavigator) {
        return (
            <main className="min-h-screen bg-slate-950 p-6 text-white">
                Loading sign-in…
            </main>
        );
    }

    if (auth.isAuthenticated && !auth.user?.expired) {
        return <Dashboard />;
    }

    const errorMessage = signInError ?? auth.error?.message;

    return (
        <main className="flex min-h-screen items-center justify-center bg-slate-950 px-4 py-8 text-white">
            <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-6">
                <p className="text-sm font-medium text-blue-400">
                    AI Operations Assistant
                </p>

                <h1 className="mt-3 text-2xl font-semibold">
                    Operations Intelligence
                </h1>

                <p className="mt-3 text-sm leading-6 text-slate-400">
                    Sign in with your approved account to open the dashboard.
                </p>

                {errorMessage && (
                    <p
                        role="alert"
                        className="mt-4 break-words rounded-lg bg-red-950 p-3 text-sm text-red-200"
                    >
                        {errorMessage}
                    </p>
                )}

                <button
                    type="button"
                    onClick={() => void signIn()}
                    className="mt-6 w-full rounded-xl bg-blue-600 px-4 py-3 text-sm font-medium transition hover:bg-blue-500"
                >
                    Sign in
                </button>
            </div>
        </main>
    );
}