// SPDX-License-Identifier: Apache-2.0
package main

import (
	"embed"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

//go:embed index.html
var webFiles embed.FS

func dashboardHandler(r *Runner, starters ...func(string, string) error) http.Handler {
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
	mux.HandleFunc("POST /api/start", func(w http.ResponseWriter, q *http.Request) {
		if origin := q.Header.Get("Origin"); origin != "" {
			u, e := url.Parse(origin)
			if e != nil || u.Host != q.Host {
				http.Error(w, "origin mismatch", http.StatusForbidden)
				return
			}
		}
		var input struct {
			Host  string `json:"host"`
			Port  int    `json:"port"`
			Token string `json:"token"`
		}
		decoder := json.NewDecoder(http.MaxBytesReader(w, q.Body, 4096))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			http.Error(w, "Enter a valid IP address and TCP port", http.StatusBadRequest)
			return
		}
		var extra any
		if decoder.Decode(&extra) != io.EOF {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		host := strings.Trim(strings.TrimSpace(input.Host), "[]")
		if (host != "localhost" && net.ParseIP(host) == nil) || input.Port < 1 || input.Port > 65535 {
			http.Error(w, "Enter an IP address and port from 1 to 65535", http.StatusBadRequest)
			return
		}
		if len(starters) == 0 {
			http.Error(w, "start unavailable", http.StatusConflict)
			return
		}
		if err := starters[0](net.JoinHostPort(host, strconv.Itoa(input.Port)), input.Token); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		w.WriteHeader(http.StatusAccepted)
	})
	mux.Handle("GET /", http.FileServer(http.FS(webFiles)))
	return mux
}
