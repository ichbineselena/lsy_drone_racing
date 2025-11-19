try:
	import jax

	# Configure jax caches if available. Wrap in try/except so tests run when
	# jax is not installed or the user doesn't want it enabled.
	jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
	jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
	jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
	# Do not enable XLA caches, crashes PyTest
	# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
except Exception:
	# jax not available or configuration failed; skip JAX-specific setup.
	pass
