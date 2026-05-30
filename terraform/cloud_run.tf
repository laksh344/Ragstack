# ── Cloud Run service ─────────────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "ragstack" {
  name     = var.service_name
  location = var.region

  template {
    service_account = google_service_account.run_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle          = true    # reduce cost when idle
        startup_cpu_boost = true    # faster cold starts
      }

      # ── Application config ─────────────────────────────────────────
      env {
        name  = "APP_HOST"
        value = "0.0.0.0"
      }
      env {
        name  = "APP_PORT"
        value = "8000"
      }
      env {
        name  = "LANGCHAIN_PROJECT"
        value = "ragstack"
      }
      env {
        name  = "LANGCHAIN_TRACING_V2"
        value = "true"
      }

      # ── Secrets from Secret Manager ────────────────────────────────
      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "LANGCHAIN_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.langchain_key.secret_id
            version = "latest"
          }
        }
      }

      ports {
        container_port = 8000
      }

      startup_probe {
        http_get { path = "/api/v1/health" }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 12
      }

      liveness_probe {
        http_get { path = "/api/v1/health" }
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_project_service.run,
    google_project_iam_member.run_sa_secret,
  ]
}

# ── Allow unauthenticated access (public demo) ────────────────────────────────
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.ragstack.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
