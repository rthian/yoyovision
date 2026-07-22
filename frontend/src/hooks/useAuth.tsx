"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { login as apiLogin } from "@/lib/api-client";
import { clearStoredToken, getStoredToken, setStoredToken } from "@/lib/auth-storage";

interface AuthContextValue {
  isAuthenticated: boolean;
  /** True until the initial localStorage read completes, so route guards
   * don't redirect-flash before we know whether a token exists. */
  isInitializing: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }): JSX.Element {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);

  useEffect(() => {
    setIsAuthenticated(getStoredToken() !== null);
    setIsInitializing(false);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const { access_token } = await apiLogin({ email, password });
    setStoredToken(access_token);
    setIsAuthenticated(true);
  }, []);

  const logout = useCallback(() => {
    clearStoredToken();
    setIsAuthenticated(false);
  }, []);

  const value = useMemo(
    () => ({ isAuthenticated, isInitializing, login, logout }),
    [isAuthenticated, isInitializing, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error("useAuth must be used within an AuthProvider.");
  }
  return context;
}
