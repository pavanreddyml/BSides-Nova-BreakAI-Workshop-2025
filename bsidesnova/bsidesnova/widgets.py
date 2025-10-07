from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
from typing import Literal, get_args, get_origin

import ipywidgets as widgets
from ipywidgets import Layout
from tensorflow.keras.utils import get_file  # <- use TF to fetch ImageNet ids/names

CATEGORY_PACKAGES: Mapping[str, str] = {
    "whitebox": "adversarial_lab.arsenal.adversarial.whitebox",
    "blackbox": "adversarial_lab.arsenal.adversarial.blackbox",
}

WHITEBOX_MODELS: Tuple[str, ...] = ("inception", "resnet", "mobilenet")
BLACKBOX_MODELS: Tuple[str, ...] = ("mnist_digits",)

MNIST_DIGITS_CLASSES = tuple(str(i) for i in range(10))


@dataclass(frozen=True)
class AttackInfo:
    """Container describing an attack implementation."""

    label: str
    cls: type


class AttackSelectorWidget:
    def __init__(self, root_path: Optional[Path | str] = None) -> None:
        self.root_path = Path(root_path or Path.cwd())
        self._attacks: Dict[str, Dict[str, AttackInfo]] = {}
        self._attack_param_widgets: List[widgets.Widget] = []
        self._imagenet_pairs_cache: Optional[List[Tuple[str, int]]] = None  # cache of [(label, idx)]

        common_style = {"description_width": "initial"}
        indent_layout = Layout(margin="0 0 0 220px")  # shift right so labels aren’t cut off

        # ---- Top: Category ----
        self.category_selector = widgets.ToggleButtons(
            options=[("Whitebox", "whitebox"), ("Blackbox", "blackbox")],
            description="Category:",
            value="whitebox",
            button_style="",
            style=common_style,
            layout=indent_layout,
        )

        # ---- Attack + params ----
        self.attack_selector = widgets.Dropdown(
            description="Attack:",
            style=common_style,
            layout=indent_layout,
        )
        self.param_container = widgets.VBox(layout=indent_layout)

        # ---- Dataset section ----
        self.dataset_header = widgets.HTML(
            value="<h4 style='margin:8px 0 0 220px;'>Dataset</h4>"
        )

        self.model_selector = widgets.Dropdown(
            description="Model:",
            style=common_style,
            layout=indent_layout,
        )

        self.image_class_selector = widgets.Dropdown(
            description="Image Class:",
            style=common_style,
            layout=indent_layout,
        )

        self.image_name_selector = widgets.Dropdown(
            description="Image Name:",
            style=common_style,
            layout=indent_layout,
        )

        # Target Class now a Dropdown with "idx - name" labels
        self.target_class_selector = widgets.Dropdown(
            description="Target Class:",
            style=common_style,
            layout=indent_layout,
        )

        self._attach_observers()
        self._initialise_state()

        # Assemble in requested order:
        self.widget = widgets.VBox(
            [
                self.category_selector,            # 1) Category
                self.attack_selector,              # 2) Attack selector
                self.param_container,              #    Attack params (rendered dynamically)
                widgets.HTML("<hr />"),
                self.dataset_header,               # 3) Dataset section
                self.model_selector,               #    Dataset
                self.image_class_selector,         #    Image Class
                self.image_name_selector,          #    Image Name
                self.target_class_selector,        #    Target Class (idx + name)
            ]
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def display(self) -> None:
        from IPython.display import display
        display(self.widget)

    def get_configuration(self) -> Dict[str, Any]:
        # Translate target_class to its numeric index (value) and include label
        target_val = self.target_class_selector.value
        target_label = dict(self.target_class_selector.options).get(target_val, None)
        return {
            "category": self.category_selector.value,
            "dataset": self.model_selector.value,  # renamed from "model"
            "attack": self.attack_selector.value,
            "attack_params": self._collect_attack_parameters(),
            "image_class": self.image_class_selector.value,
            "image_name": self.image_name_selector.value,
            "target_class_idx": target_val,
            "target_class_label": target_label,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _attach_observers(self) -> None:
        self.category_selector.observe(self._on_category_change, names="value")
        self.model_selector.observe(self._on_model_change, names="value")
        self.attack_selector.observe(self._on_attack_change, names="value")
        self.image_class_selector.observe(self._on_image_class_change, names="value")

    def _initialise_state(self) -> None:
        self._refresh_models()
        self._refresh_attacks()
        self._refresh_datasets()

    def _on_category_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        self._refresh_models()
        self._refresh_attacks()
        self._imagenet_pairs_cache = None  # reset on category switch
        self._refresh_datasets()

    def _on_model_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        self._refresh_datasets()

    def _on_attack_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        self._render_attack_parameters()

    def _on_image_class_change(self, change: MutableMapping[str, Any]) -> None:
        if change.get("new") == change.get("old"):
            return
        self._refresh_image_names()

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def _refresh_models(self) -> None:
        category = self.category_selector.value
        if category == "whitebox":
            options = [(model.title(), model) for model in WHITEBOX_MODELS]
        else:
            options = [(model.replace("_", " ").title(), model) for model in BLACKBOX_MODELS]

        self.model_selector.options = options
        self.model_selector.value = options[0][1] if options else None

    def _dataset_root(self) -> Optional[Path]:
        category = self.category_selector.value
        if category == "whitebox":
            candidate = self._first_existing(
                self.root_path / "assets" / "images" / "imagenet",
                self.root_path / "examples" / "data" / "imagenet",
            )
        else:
            model = self.model_selector.value or ""
            if model == "mnist_digits":
                candidate = self._first_existing(
                    self.root_path / "assets" / "images" / "digits",
                    self.root_path / "examples" / "data" / "digits",
                )
            else:
                candidate = None
        if candidate and candidate.exists():
            return candidate
        return None

    def _refresh_datasets(self) -> None:
        dataset_root = self._dataset_root()
        classes = self._list_directory_names(dataset_root)

        # Image class dropdown (directory names if present)
        class_options = self._build_options(classes, empty_label="No classes available")
        self.image_class_selector.options = class_options
        if len(class_options) > 1:
            self.image_class_selector.value = class_options[1][1]
        elif class_options:
            self.image_class_selector.value = class_options[0][1]
        else:
            self.image_class_selector.value = None

        # Target class dropdown (idx + label from ImageNet via TF, or MNIST)
        target_idx_options = self._target_class_options(category=self.category_selector.value)
        self.target_class_selector.options = target_idx_options
        self.target_class_selector.value = (
            target_idx_options[1][1] if len(target_idx_options) > 1 else (target_idx_options[0][1] if target_idx_options else None)
        )

        self._refresh_image_names()

    def _refresh_image_names(self) -> None:
        dataset_root = self._dataset_root()
        class_name = self.image_class_selector.value
        image_dir = dataset_root / class_name if dataset_root and class_name else None
        image_names = self._list_file_names(image_dir)

        image_options = self._build_options(image_names, empty_label="No images available")
        self.image_name_selector.options = image_options
        if len(image_options) > 1:
            self.image_name_selector.value = image_options[1][1]
        elif image_options:
            self.image_name_selector.value = image_options[0][1]
        else:
            self.image_name_selector.value = None

    def _target_class_options(self, category: str) -> List[Tuple[str, Optional[int]]]:
        """Return dropdown options for Target Class as [(label, idx), ...]."""
        if category == "whitebox":
            pairs = self._imagenet_idx_label_pairs_from_tf()
        else:
            model = self.model_selector.value or ""
            if model == "mnist_digits":
                pairs = [(f"{name} - {i}", i) for i, name in enumerate(MNIST_DIGITS_CLASSES)]
            else:
                pairs = []

        if not pairs:
            return [("No target classes available", None)]
        return [("Select...", None)] + pairs

    def _imagenet_idx_label_pairs_from_tf(self) -> List[Tuple[str, int]]:
        """Load ImageNet (ILSVRC-2012) class ids/names via TensorFlow Keras.

        Uses keras.utils.get_file to fetch the canonical JSON used by
        tf.keras.applications.imagenet_utils.decode_predictions.
        """
        if self._imagenet_pairs_cache is not None:
            return self._imagenet_pairs_cache

        # Download (or use cached) class index JSON via TF utility
        json_path = get_file(
            fname="imagenet_class_index.json",
            origin="https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
            cache_subdir="models",
            file_hash=None,
        )
        with open(json_path, "r") as f:
            class_index: Dict[str, List[str]] = json.load(f)  # e.g., {"0": ["n01440764","tench"],...}

        # Build pairs sorted by integer index
        items: List[Tuple[str, int]] = []
        for k, (wnid, name) in class_index.items():
            idx = int(k)
            label = f"{name} - {idx}"
            items.append((label, idx))
        items.sort(key=lambda x: x[1])

        self._imagenet_pairs_cache = items
        return items

    # ------------------------------------------------------------------
    # Attack helpers
    # ------------------------------------------------------------------
    def _refresh_attacks(self) -> None:
        category = self.category_selector.value
        attacks = self._discover_attacks(category)
        options = self._build_options(attacks.keys(), empty_label="No attacks found")
        self.attack_selector.options = options
        if len(options) > 1:
            self.attack_selector.value = options[1][1]
        elif options:
            self.attack_selector.value = options[0][1]
        else:
            self.attack_selector.value = None
        self._render_attack_parameters()

    def _render_attack_parameters(self) -> None:
        selected = self.attack_selector.value
        category = self.category_selector.value
        attack_info = self._attacks.get(category, {}).get(selected)

        if not attack_info:
            self.param_container.children = [widgets.HTML("<em>No attack selected.</em>")]
            self._attack_param_widgets = []
            return

        widgets_list = self._create_widgets_for_attack(attack_info.cls)
        self._attack_param_widgets = widgets_list
        if widgets_list:
            self.param_container.children = widgets_list
        else:
            self.param_container.children = [widgets.HTML("<em>No configurable parameters.</em>")]

    def _collect_attack_parameters(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for widget in self._attack_param_widgets:
            name = getattr(widget, "_param_name", None)
            if not name:
                continue
            value = getattr(widget, "value", None)
            params[name] = value
        return params

    def _discover_attacks(self, category: str) -> Dict[str, AttackInfo]:
        if category in self._attacks:
            return self._attacks[category]

        module_name = CATEGORY_PACKAGES[category]
        module = importlib.import_module(module_name)
        attack_infos: Dict[str, AttackInfo] = {}

        exported_names = getattr(module, "__all__", None)
        if not exported_names:
            exported_names = [name for name, obj in inspect.getmembers(module, inspect.isclass)]

        for name in exported_names:
            cls = getattr(module, name, None)
            if not inspect.isclass(cls):
                continue
            label = self._humanize_name(name)
            attack_infos[label] = AttackInfo(label=label, cls=cls)

        self._attacks[category] = attack_infos
        return attack_infos

    def _create_widgets_for_attack(self, attack_cls: type) -> List[widgets.Widget]:
        signature = inspect.signature(attack_cls.__init__)
        widgets_list: List[widgets.Widget] = []
        for name, parameter in signature.parameters.items():
            # Exclude certain parameters globally
            if name in {"self", "model", "preprocess_fn", "pred_fn"}:
                continue
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            widget = self._widget_for_parameter(name, parameter)
            if widget is None:
                continue
            widget._param_name = name  # type: ignore[attr-defined]
            widgets_list.append(widget)
        return widgets_list

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _humanize_name(name: str) -> str:
        spaced = []
        previous_lower = False
        for ch in name:
            if ch.isupper() and previous_lower:
                spaced.append(" ")
            spaced.append(ch)
            previous_lower = ch.islower()
        human = "".join(spaced)
        return human.replace("Attack", "").strip()

    @staticmethod
    def _build_options(values: Iterable[str], empty_label: str) -> List[Tuple[str, Optional[str]]]:
        items = sorted(set(filter(None, values)))
        if not items:
            return [(empty_label, None)]
        options = [("Select...", None)]
        options.extend([(item, item) for item in items])
        return options

    @staticmethod
    def _list_directory_names(path: Optional[Path]) -> List[str]:
        if not path or not path.exists():
            return []
        return [child.name for child in path.iterdir() if child.is_dir()]

    @staticmethod
    def _list_file_names(path: Optional[Path]) -> List[str]:
        if not path or not path.exists():
            return []
        return [child.name for child in path.iterdir() if child.is_file()]

    @staticmethod
    def _first_existing(*candidates: Path) -> Optional[Path]:
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    @staticmethod
    def _widget_for_parameter(name: str, parameter: inspect.Parameter) -> Optional[widgets.Widget]:
        # Unified style/layout to avoid cut-off labels
        common_style = {"description_width": "initial"}
        indent_layout = Layout(margin="0 0 0 220px")

        annotation = parameter.annotation
        default = parameter.default
        description = name.replace("_", " ").title()

        origin = get_origin(annotation)
        if origin is not None:
            if origin in {list, tuple, set} or (
                isinstance(origin, type) and issubclass(origin, IterableABC)
            ):
                return widgets.Text(description=description, placeholder="Comma separated list", style=common_style, layout=indent_layout)
            if origin is Literal:
                literals = list(get_args(annotation))
                options = literals if literals else [default]
                options = [opt for opt in options if opt is not inspect._empty]
                return widgets.Dropdown(
                    description=description,
                    options=options,
                    value=default if default in options else (options[0] if options else None),
                    style=common_style,
                    layout=indent_layout,
                )
            args = get_args(annotation)
            if args:
                unwrapped = [arg for arg in args if arg is not type(None)]  # noqa: E721
                if len(unwrapped) == 1:
                    annotation = unwrapped[0]

        if default is inspect._empty:
            return widgets.Text(description=description, placeholder="Required", style=common_style, layout=indent_layout)

        if isinstance(default, bool):
            return widgets.Checkbox(description=description, value=default, style=common_style, layout=indent_layout)
        if isinstance(default, int):
            return widgets.IntText(description=description, value=default, style=common_style, layout=indent_layout)
        if isinstance(default, float):
            return widgets.FloatText(description=description, value=default, style=common_style, layout=indent_layout)
        if isinstance(default, str):
            return widgets.Text(description=description, value=default, style=common_style, layout=indent_layout)
        if default is None:
            return widgets.Text(description=description, placeholder="None", style=common_style, layout=indent_layout)

        return widgets.Text(description=description, value=str(default), style=common_style, layout=indent_layout)
