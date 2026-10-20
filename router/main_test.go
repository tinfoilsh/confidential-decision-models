package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestBackend returns a backend proxying to a stub server that records
// the request and answers with its own name.
func newTestBackend(t *testing.T, name string, lastBody *string) (*backend, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data":   []map[string]any{{"id": name, "object": "model"}},
			})
		case "/metrics":
			_, _ = io.WriteString(w, "# HELP vllm_fake fake\n# TYPE vllm_fake counter\nvllm_fake{model_name=\""+name+"\"} 1\n")
		default:
			body, _ := io.ReadAll(r.Body)
			if lastBody != nil {
				*lastBody = string(body)
			}
			_, _ = io.WriteString(w, name)
		}
	}))
	t.Cleanup(srv.Close)

	backends, err := buildBackends(map[string]string{name: srv.URL})
	if err != nil {
		t.Fatalf("buildBackends: %v", err)
	}
	return backends[name], srv
}

func newTestHandler(t *testing.T, lastBody *string) http.Handler {
	t.Helper()
	backends := map[string]*backend{}
	for _, name := range []string{"d3", "d3-flash", "d3-mini"} {
		b, _ := newTestBackend(t, name, lastBody)
		backends[name] = b
	}
	return newHandler(backends)
}

func TestRoutesOnBodyModel(t *testing.T) {
	var lastBody string
	h := newTestHandler(t, &lastBody)

	for _, model := range []string{"d3", "d3-flash", "d3-mini"} {
		body := `{"model":"` + model + `","state":"s","questions":{"q":{"type":"noul","instructions":"x?"}}}`
		req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status = %d", model, rec.Code)
		}
		if got := rec.Body.String(); got != model {
			t.Fatalf("%s: routed to %q", model, got)
		}
		if lastBody != body {
			t.Fatalf("%s: body not replayed intact: %q", model, lastBody)
		}
	}
}

func TestUnknownModel(t *testing.T) {
	h := newTestHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"gpt-4"}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "d3-mini") {
		t.Fatalf("error should list available models: %q", rec.Body.String())
	}
}

func TestNonJSONBody(t *testing.T) {
	h := newTestHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader("not json"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

func TestMergedModels(t *testing.T) {
	h := newTestHandler(t, nil)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var out struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out.Data) != 3 {
		t.Fatalf("merged %d models, want 3", len(out.Data))
	}
}

func TestMergedMetricsStripsComments(t *testing.T) {
	h := newTestHandler(t, nil)
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	body := rec.Body.String()
	if strings.Contains(body, "#") {
		t.Fatalf("comments not stripped: %q", body)
	}
	for _, name := range []string{"d3", "d3-flash", "d3-mini"} {
		if !strings.Contains(body, `model_name="`+name+`"`) {
			t.Fatalf("missing series for %s: %q", name, body)
		}
	}
}

func TestHealth(t *testing.T) {
	h := newTestHandler(t, nil)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || rec.Body.String() != "ok" {
		t.Fatalf("health = %d %q", rec.Code, rec.Body.String())
	}
}
