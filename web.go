// SPDX-License-Identifier: Apache-2.0
package main

import (
	"embed"
	"encoding/json"
	"net/http"
	"net/url"
)

//go:embed index.html
var webFiles embed.FS

func dashboardHandler(r *Runner) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/state", func(w http.ResponseWriter, q *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_ = json.NewEncoder(w).Encode(r.snapshot())
	})
	for _, action := range []string{"pause", "resume"} {
		pause := action == "pause"
		mux.HandleFunc("POST /api/"+action, func(w http.ResponseWriter, q *http.Request) {
			if origin := q.Header.Get("Origin"); origin != "" {
				u, e := url.Parse(origin)
				if e != nil || u.Host != q.Host {
					http.Error(w, "origin mismatch", http.StatusForbidden)
					return
				}
			}
			r.paused.Store(pause)
			w.WriteHeader(http.StatusNoContent)
		})
	}
	mux.Handle("GET /", http.FileServer(http.FS(webFiles)))
	return mux
}
