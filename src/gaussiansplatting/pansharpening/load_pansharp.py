from .algorithm import ihs_fusion, simple_brovey, brovey_pansharp


def load_pansharp(pansharp_cfg: "DictConfig"):
    method = pansharp_cfg.method
    if method == "ihs":
        return ihs_fusion

    elif method == "simple_brovey":
        return simple_brovey
    elif method == "brovey":
        return brovey_pansharp
    elif method == "nopansharp":
        return None
    else:
        raise ValueError(
            f"Unsupported pansharpening method: {method}. Supported methods are 'ihs' and 'brovey'."
        )
