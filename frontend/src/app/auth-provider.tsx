"use client";

import { useEffect, useState, type ReactNode } from "react";
import { AuthProvider } from "react-oidc-context";
import { WebStorageStateStore } from "oidc-client-ts";

export default function CognitoAuthProvider({
    children,
}: {
    children: ReactNode;
}) {
    const [ready, setReady] = useState(false);

    useEffect(() => {
        setReady(true);
    }, []);

    // Browser storage is available only after the page mounts.
    if (!ready) {
        return (
            <div className="min-h-screen bg-slate-950 p-6 text-white">
                Loading sign-in…
            </div>
        );
    }

    return (
        <AuthProvider
            authority="https://cognito-idp.eu-west-1.amazonaws.com/eu-west-1_c5VlvzE6e"
            client_id="5bjde7k6tj65g17kk96vdqpert"
            redirect_uri={`${window.location.origin}/auth/callback`}
            response_type="code"
            scope="openid email profile"
            automaticSilentRenew={true}
            userStore={
                new WebStorageStateStore({
                    store: window.sessionStorage,
                })
            }
            onSigninCallback={() => {
                window.location.replace("/");
            }}
        >
            {children}
        </AuthProvider>
    );
}