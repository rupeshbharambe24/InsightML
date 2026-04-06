"""ModelRegistry — register, unregister, and filter model entries."""

from __future__ import annotations

from insightml.battle.catalog import MODEL_CATALOG, ModelEntry


class ModelRegistry:
    """Registry of model entries, supporting custom registration and filtering.

    Usage::

        registry = ModelRegistry()
        registry.available("classification")     # list of ModelEntry
        registry.register(my_model_entry)        # add custom model
        registry.unregister("RandomForest")      # remove a model
        registry.filter(families=["tree"])       # subset by family
        registry.get("RandomForestClassifier")   # single entry
    """

    def __init__(self, copy_catalog: bool = True) -> None:
        self._catalog: dict[str, ModelEntry] = (
            dict(MODEL_CATALOG) if copy_catalog else MODEL_CATALOG
        )

    # --- Query ---

    def available(self, task: str, include_optional: bool = True) -> list[ModelEntry]:
        """Return all available (importable) model entries for a task.

        Args:
            task: "classification" or "regression".
            include_optional: If True, include optional extras that are installed.

        Returns:
            List of ModelEntry sorted by family then name.
        """
        # Use the live catalog (which may have custom entries)
        entries = [
            e for e in self._catalog.values()
            if e.task in (task, "both")
            and (not e.is_optional or (include_optional and e.is_available()))
        ]
        return sorted(entries, key=lambda e: (e.family, e.name))

    def get(self, name: str) -> ModelEntry:
        """Return a single ModelEntry by name.

        Raises:
            KeyError: If model name is not in the registry.
        """
        if name not in self._catalog:
            raise KeyError(
                f"Model '{name}' not in registry. "
                f"Available: {list(self._catalog.keys())}"
            )
        return self._catalog[name]

    def filter(
        self,
        task: str | None = None,
        families: list[str] | None = None,
        names: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[ModelEntry]:
        """Filter models by task, family, explicit names, or exclusions.

        Args:
            task: If given, filter to this task.
            families: If given, only include these families (e.g. ["tree", "linear"]).
            names: If given, only include these specific model names.
            exclude: If given, exclude these model names.

        Returns:
            Filtered list of ModelEntry.
        """
        entries = list(self._catalog.values())

        if task:
            entries = [e for e in entries if e.task in (task, "both")]
        if families:
            entries = [e for e in entries if e.family in families]
        if names:
            entries = [e for e in entries if e.name in names]
        if exclude:
            entries = [e for e in entries if e.name not in exclude]

        return [e for e in entries if e.is_available()]

    def names(self, task: str | None = None) -> list[str]:
        """Return list of all model names, optionally filtered by task."""
        if task:
            return [e.name for e in self._catalog.values()
                    if e.task in (task, "both")]
        return list(self._catalog.keys())

    # --- Mutation ---

    def register(self, entry: ModelEntry) -> None:
        """Add or replace a model in the registry.

        Args:
            entry: ModelEntry to register.
        """
        self._catalog[entry.name] = entry

    def unregister(self, name: str) -> None:
        """Remove a model from the registry.

        Args:
            name: Name of the model to remove.

        Raises:
            KeyError: If the model is not registered.
        """
        if name not in self._catalog:
            raise KeyError(f"Model '{name}' not registered.")
        del self._catalog[name]

    def __repr__(self) -> str:
        n_clf = sum(1 for e in self._catalog.values()
                    if e.task in ("classification", "both"))
        n_reg = sum(1 for e in self._catalog.values()
                    if e.task in ("regression", "both"))
        return f"ModelRegistry({n_clf} classifiers, {n_reg} regressors)"


# Module-level default registry
_default_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Return the module-level default registry."""
    return _default_registry
