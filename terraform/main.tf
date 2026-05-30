terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  # Uncomment to use GCS backend (recommended for team use)
  # backend "gcs" {
  #   bucket = "<your-tf-state-bucket>"
  #   prefix = "ragstack/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Enable required APIs ──────────────────────────────────────────────────────
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifact_registry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secret_manager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

# ── Artifact Registry for Docker images ──────────────────────────────────────
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "ragstack"
  format        = "DOCKER"
  description   = "RAGStack container images"

  depends_on = [google_project_service.artifact_registry]
}

# ── Secret Manager — API keys ─────────────────────────────────────────────────
resource "google_secret_manager_secret" "openai_key" {
  secret_id = "ragstack-openai-api-key"
  replication { auto {} }
  depends_on = [google_project_service.secret_manager]
}

resource "google_secret_manager_secret_version" "openai_key" {
  secret      = google_secret_manager_secret.openai_key.id
  secret_data = var.openai_api_key
}

resource "google_secret_manager_secret" "langchain_key" {
  secret_id = "ragstack-langchain-api-key"
  replication { auto {} }
  depends_on = [google_project_service.secret_manager]
}

resource "google_secret_manager_secret_version" "langchain_key" {
  secret      = google_secret_manager_secret.langchain_key.id
  secret_data = var.langchain_api_key
}

# ── Service account for Cloud Run ─────────────────────────────────────────────
resource "google_service_account" "run_sa" {
  account_id   = "ragstack-run-sa"
  display_name = "RAGStack Cloud Run Service Account"
}

resource "google_project_iam_member" "run_sa_secret" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.run_sa.email}"
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.ragstack.uri
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/ragstack"
}
