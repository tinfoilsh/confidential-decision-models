// Package main implements the Tinfoil decision-models reverse proxy.
//
// Three structured-decision models (the vllm-sr d3 family) share this
// enclave: decision-large, decision-medium, and decision-small. The router dispatches
// each request on the "model" field of its JSON body.
//
//	POST /v1/decisions {"model":"decision-small",...} -> decision-small container
//	GET  /v1/models                            -> merged list from all backends
//	GET  /metrics                              -> concatenated backend metrics
//	GET  /health                               -> router's own health
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"
)

// maxBodyBytes caps /v1/decisions request bodies, and thereby the router's
// per-request memory (the body is buffered to read its "model" field). The
// engines serve 32k-token contexts (~130 KB of text), so 4 MiB is ~30x the
// largest legitimate request. Revisit if image inputs are ever enabled.
const maxBodyBytes = 4 << 20

type backend struct {
	name     string
	upstream *url.URL
	proxy    *httputil.ReverseProxy
}

func main() {
	listenAddr := getenvDefault("LISTEN_ADDR", ":8080")
	specs := map[string]string{
		"decision-large": getenvDefault("DECISION_LARGE_URL", "http://127.0.0.1:8103"),
		"decision-medium":  getenvDefault("DECISION_MEDIUM_URL", "http://127.0.0.1:8102"),
		"decision-small":  getenvDefault("DECISION_SMALL_URL", "http://127.0.0.1:8101"),
	}

	backends, err := buildBackends(specs)
	if err != nil {
		log.Fatalf("failed to build router: %v", err)
	}

	server := &http.Server{
		Addr:              listenAddr,
		Handler:           newHandler(backends),
		ReadHeaderTimeout: 15 * time.Second,
	}

	log.Printf("starting decision-models router on %s", listenAddr)
	for name, raw := range specs {
		log.Printf("backend %-8s -> %s", name, raw)
	}

	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server failed: %v", err)
	}
}

func newHandler(backends map[string]*backend) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = w.Write([]byte("ok"))
	})

	mux.HandleFunc("/v1/decisions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		serveDecision(w, r, backends)
	})

	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		serveMergedModels(w, r, backends)
	})

	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		serveMergedMetrics(w, r, backends)
	})

	return mux
}

// serveDecision reads the JSON body, routes on its "model" field, and
// replays the body to the selected backend.
func serveDecision(w http.ResponseWriter, r *http.Request, backends map[string]*backend) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	if len(body) > maxBodyBytes {
		http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
		return
	}

	var probe struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		http.Error(w, "request body must be JSON", http.StatusBadRequest)
		return
	}

	b, ok := backends[probe.Model]
	if !ok {
		http.Error(w, fmt.Sprintf("unknown model: %q (available: %s)", probe.Model, modelList(backends)), http.StatusNotFound)
		return
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Del("Content-Length") // ReverseProxy re-derives it from ContentLength
	b.proxy.ServeHTTP(w, r)
}

// serveMergedModels aggregates every backend's /v1/models into one OpenAI
// model list, so clients can discover all three models from one hostname.
func serveMergedModels(w http.ResponseWriter, r *http.Request, backends map[string]*backend) {
	type modelEntry = map[string]any
	merged := struct {
		Object string       `json:"object"`
		Data   []modelEntry `json:"data"`
	}{Object: "list"}

	client := &http.Client{Timeout: 10 * time.Second}
	for _, name := range sortedNames(backends) {
		b := backends[name]
		resp, err := client.Get(b.upstream.String() + "/v1/models")
		if err != nil {
			log.Printf("models fetch failed for %s: %v", b.name, err)
			continue
		}
		var out struct {
			Data []modelEntry `json:"data"`
		}
		err = json.NewDecoder(resp.Body).Decode(&out)
		_ = resp.Body.Close()
		if err != nil {
			log.Printf("models decode failed for %s: %v", b.name, err)
			continue
		}
		merged.Data = append(merged.Data, out.Data...)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(merged)
}

// serveMergedMetrics concatenates the backends' Prometheus expositions.
// Comment lines are stripped so duplicate HELP/TYPE pairs can't produce an
// invalid exposition; series stay distinct via their model_name labels.
func serveMergedMetrics(w http.ResponseWriter, r *http.Request, backends map[string]*backend) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	client := &http.Client{Timeout: 10 * time.Second}
	for _, name := range sortedNames(backends) {
		b := backends[name]
		resp, err := client.Get(b.upstream.String() + "/metrics")
		if err != nil {
			log.Printf("metrics fetch failed for %s: %v", b.name, err)
			continue
		}
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			log.Printf("metrics read failed for %s: %v", b.name, err)
			continue
		}
		for _, line := range strings.Split(string(body), "\n") {
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			_, _ = io.WriteString(w, line+"\n")
		}
	}
}

func buildBackends(specs map[string]string) (map[string]*backend, error) {
	backends := make(map[string]*backend, len(specs))
	for name, raw := range specs {
		target, err := url.Parse(raw)
		if err != nil {
			return nil, fmt.Errorf("parse %s url: %w", name, err)
		}
		if target.Scheme == "" || target.Host == "" {
			return nil, fmt.Errorf("%s url must include scheme and host: %q", name, raw)
		}

		proxy := httputil.NewSingleHostReverseProxy(target)
		director := proxy.Director
		targetHost := target.Host
		backendName := name
		proxy.Director = func(req *http.Request) {
			director(req)
			req.Host = targetHost
		}
		proxy.ErrorLog = log.New(os.Stderr, fmt.Sprintf("%s proxy: ", name), log.LstdFlags)
		proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("proxy error for %s %s via %s: %v", r.Method, r.URL.Path, backendName, err)
			http.Error(w, "bad gateway", http.StatusBadGateway)
		}

		backends[name] = &backend{name: name, upstream: target, proxy: proxy}
	}
	return backends, nil
}

func sortedNames(backends map[string]*backend) []string {
	names := make([]string, 0, len(backends))
	for name := range backends {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func modelList(backends map[string]*backend) string {
	return strings.Join(sortedNames(backends), ", ")
}

func getenvDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
