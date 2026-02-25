import csv
import hashlib
import io
import json
import shutil
import sqlite3
from pathlib import Path
import sys
import urllib.request
import zipfile

import h5py
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
LOSCHMIDT_CM3 = 2.6867805e19
BOLTZMANN_CGS = 1.380649e-16
SMALL = 1.0e-300


def _safe_extract_zip(zip_file: zipfile.ZipFile, extract_to: Path) -> None:
    extract_to = extract_to.resolve()
    for member in zip_file.infolist():
        member_path = (extract_to / member.filename).resolve()
        if extract_to not in member_path.parents and member_path != extract_to:
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    zip_file.extractall(extract_to)


def download_and_extract_zip(
    url: str,
    extract_to: str | Path,
    zip_path: str | Path | None = None,
) -> Path:
    """Download a zip file from `url` and extract it into `extract_to`."""
    extract_dir = Path(extract_to)
    extract_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    done_marker = extract_dir / f".download_extract_done_{url_hash}"
    legacy_done_marker = extract_dir / ".download_extract_done"

    if done_marker.exists():
        return extract_dir
    if legacy_done_marker.exists():
        marker_text = legacy_done_marker.read_text(encoding="utf-8")
        if f"url={url}\n" in marker_text:
            return extract_dir

    zip_file_path = Path(zip_path) if zip_path is not None else extract_dir / "download.zip"
    zip_file_path.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\rDownloading {zip_file_path.name}: {percent:6.2f}% "
                f"({downloaded_mb:,.2f}/{total_mb:,.2f} MB)"
            )
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            sys.stdout.write(
                f"\rDownloading {zip_file_path.name}: {downloaded_mb:,.2f} MB"
            )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, zip_file_path, reporthook=_reporthook)
    sys.stdout.write("\n")
    sys.stdout.flush()

    with zipfile.ZipFile(zip_file_path, "r") as zf:
        _safe_extract_zip(zf, extract_dir)

    done_marker.write_text(f"url={url}\nzip={zip_file_path}\n", encoding="utf-8")

    return extract_dir


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (CURRENT_DIR.parent / p).resolve()
    return repo_candidate


def _source_fingerprint(paths: list[Path], extra: dict | None = None) -> dict:
    files = []
    for path in sorted(paths):
        st = path.stat()
        files.append({"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    out = {"files": files}
    if extra is not None:
        out["extra"] = extra
    return out


def _marker_matches(marker: Path, fingerprint: dict) -> bool:
    if not marker.exists():
        return False
    try:
        prev = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return prev == fingerprint


def _write_marker(marker: Path, fingerprint: dict) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")


def _decode_numpy_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _wave_edges_um_from_wno(wno: np.ndarray, delta_wno: np.ndarray) -> np.ndarray:
    left = wno - (delta_wno / 2.0)
    right = wno + (delta_wno / 2.0)
    wno_edges = np.concatenate(([left[0]], right))
    if np.any(wno_edges <= 0.0):
        raise ValueError("Encountered non-positive wavenumber edge while building wavelength bins.")
    return (1.0e4 / wno_edges)[::-1]


def _write_h5_kdistribution(
    out_file: Path,
    species: str,
    wavelengths: np.ndarray,
    temp: np.ndarray,
    log10p: np.ndarray,
    weights: np.ndarray,
    log10k: np.ndarray,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        f.create_dataset("species", data=np.bytes_(species))
        f.create_dataset("wavelengths", data=wavelengths.astype(np.float32))
        f.create_dataset("T", data=temp.astype(np.float32))
        f.create_dataset("log10P", data=log10p.astype(np.float32))
        f.create_dataset("weights", data=weights.astype(np.float32))
        f.create_dataset("log10k", data=log10k.astype(np.float32))
        note = (
            "Converted from PICASO *_1460.hdf5 into photochem/clima k-distribution schema. "
            "Units: log10k = log10(cm^2/molecule), wavelengths = um, T = K, log10P = log10(bar)."
        )
        f.create_dataset("notes", data=np.bytes_(note))


def _write_h5_cia(out_file: Path, wavelengths: np.ndarray, temp: np.ndarray, log10xs: np.ndarray, notes: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        f.create_dataset("wavelengths", data=wavelengths.astype(np.float32))
        f.create_dataset("T", data=temp.astype(np.float32))
        f.create_dataset("log10xs", data=log10xs.astype(np.float32))
        f.create_dataset("notes", data=np.bytes_(notes))


def _write_h5_photolysis_xsection(
    out_file: Path,
    wavelengths: np.ndarray,
    photoabsorption: np.ndarray,
    photodissociation: np.ndarray,
    photoionisation: np.ndarray,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        f.create_dataset("wavelengths", data=wavelengths.astype(np.float32))
        f.create_dataset("photoabsorption", data=photoabsorption.astype(np.float32))
        f.create_dataset("photodissociation", data=photodissociation.astype(np.float32))
        f.create_dataset("photoionisation", data=photoionisation.astype(np.float32))


def convert_kdistributions(
    picaso_downloads: Path,
    output_dir: Path,
    marker_dir: Path,
    *,
    force: bool = False,
) -> tuple[int, np.ndarray]:
    src_files = sorted(picaso_downloads.glob("*_1460.hdf5"))
    if not src_files:
        raise FileNotFoundError(f"No PICASO k-distribution files found in: {picaso_downloads}")

    marker = marker_dir / "kdistributions.done"
    fingerprint = _source_fingerprint(src_files)
    with h5py.File(src_files[0], "r") as f0:
        reference_wno = f0["wno"][:]

    if not force and _marker_matches(marker, fingerprint):
        return len(src_files), reference_wno

    kout = output_dir / "kdistributions"
    kout.mkdir(parents=True, exist_ok=True)

    first_wavelengths = None
    first_weights = None

    for src in src_files:
        species = src.name.replace("_1460.hdf5", "")
        with h5py.File(src, "r") as f:
            kcoeffs = f["kcoeffs"][:]  # (P,T,W,G)
            pressures = f["pressures"][:]
            temperatures = f["temperatures"][:]
            wno = f["wno"][:]
            delta_wno = f["delta_wno"][:]
            gauss_wts = f["gauss_wts"][:]

        p_axis = np.unique(pressures)
        t_axis = np.unique(temperatures)
        if p_axis.size != kcoeffs.shape[0] or t_axis.size != kcoeffs.shape[1]:
            raise ValueError(f"Unexpected P/T layout in {src}")

        wavelengths = _wave_edges_um_from_wno(wno, delta_wno)

        # PICASO stores ln(k). Convert to log10(k): log10(k)=ln(k)/ln(10).
        log10k = (kcoeffs / np.log(10.0)).transpose(2, 1, 0, 3)
        log10k = log10k[::-1, :, :, :]

        log10p = np.log10(p_axis)
        _write_h5_kdistribution(
            kout / f"{species}.h5",
            species,
            wavelengths,
            t_axis,
            log10p,
            gauss_wts,
            log10k,
        )

        if first_wavelengths is None:
            first_wavelengths = wavelengths
            first_weights = gauss_wts
        else:
            if not np.allclose(first_wavelengths, wavelengths, rtol=0.0, atol=1e-8):
                raise ValueError(f"Wavelength bins mismatch in {src.name}")
            if not np.allclose(first_weights, gauss_wts, rtol=0.0, atol=1e-12):
                raise ValueError(f"Gauss weights mismatch in {src.name}")

    bins_file = kout / "bins.h5"
    with h5py.File(bins_file, "w") as f:
        # Per request, no split: both channels cover full wavelength range.
        f.create_dataset("sol_wavl", data=first_wavelengths.astype(np.float32))
        f.create_dataset("ir_wavl", data=first_wavelengths.astype(np.float32))

    _write_marker(marker, fingerprint)
    return len(src_files), reference_wno


def _load_h2minus_table(csv_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with csv_file.open("r", encoding="utf-8") as f:
        for _ in range(5):
            next(f)
        reader = csv.reader(f)
        header = next(reader)
        lambda_angstrom = np.array([float(x) for x in header[1:]], dtype=float)
        theta_vals = []
        data_rows = []
        for row in reader:
            theta_vals.append(float(row[0]))
            data_rows.append([float(x) for x in row[1:]])
    theta = np.array(theta_vals, dtype=float)
    table = np.array(data_rows, dtype=float)
    # Bell+1980 table columns are wavelength in Angstrom. Convert to wavenumber cm^-1.
    wno_bell = 1.0e8 / lambda_angstrom
    return theta, wno_bell, table


def _get_h2minus(t: float, new_wno: np.ndarray, theta: np.ndarray, wno_bell: np.ndarray, table: np.ndarray) -> np.ndarray:
    new_theta = 5040.0 / t
    idx = int(np.argmin(np.abs(theta - new_theta)))
    # table units are 10^26 * cm4/dyn, so multiply by 1e-26
    kappa_bell = table[idx, :] * 1.0e-26
    return np.interp(new_wno, wno_bell, kappa_bell, left=1.0e-33, right=1.0e-33)


def _get_hminusbf(wno: np.ndarray) -> np.ndarray:
    coeff = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982], dtype=float)[::-1]
    lambda_0 = 1.6419
    wave = 1.0e4 / wno
    nonzero = np.where(wno > 1.0e4 / lambda_0)
    f = np.zeros(wave.size)
    x = np.zeros(wave.size)
    result = np.zeros(wave.size) + 1.0e-33
    x[nonzero] = np.sqrt(1.0 / wave[nonzero] - 1.0 / lambda_0)
    for c in coeff:
        f[nonzero] = f[nonzero] * x[nonzero] + c
    result[nonzero] = (wave[nonzero] * x[nonzero]) ** 3 * f[nonzero] * 1.0e-18
    return result


def _get_hminusff(t: float, wno: np.ndarray) -> np.ndarray:
    AJ1 = [0.0, 2483.346, -3449.889, 2200.040, -696.271, 88.283]
    BJ1 = [0.0, 285.827, -1158.382, 2427.719, -1841.400, 444.517]
    CJ1 = [0.0, -2054.291, 8746.523, -13651.105, 8624.970, -1863.864]
    DJ1 = [0.0, 2827.776, -11485.632, 16755.524, -10051.530, 2095.288]
    EJ1 = [0.0, -1341.537, 5303.609, -7510.494, 4400.067, -901.788]
    FJ1 = [0.0, 208.952, -812.939, 1132.738, -655.020, 132.985]
    AJ2 = [518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0]
    BJ2 = [-734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0]
    CJ2 = [1021.1775, -1977.3395, 1096.8827, -245.649, 0.0, 0.0]
    DJ2 = [-479.0721, 922.3575, -521.1341, 114.243, 0.0, 0.0]
    EJ2 = [93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0]
    FJ2 = [-6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0]

    wave = 1.0e4 / wno
    nwave = wave.size
    if t < 800.0:
        return np.zeros(nwave) + 1.0e-60

    t_coeff = 5040.0 / t
    hj = np.zeros((6, nwave))
    longw = np.where(wave > 0.3645)
    midw = np.where(wave <= 0.3645)
    shortw = np.where(wave < 0.1823)
    wave = wave.copy()
    wave[shortw] = 0.1823

    for i in range(6):
        hj[i, longw] = 1.0e-29 * (
            wave[longw] * wave[longw] * AJ1[i]
            + BJ1[i]
            + (CJ1[i] + (DJ1[i] + (EJ1[i] + FJ1[i] / wave[longw]) / wave[longw]) / wave[longw]) / wave[longw]
        )
        hj[i, midw] = 1.0e-29 * (
            wave[midw] * wave[midw] * AJ2[i]
            + BJ2[i]
            + (CJ2[i] + (DJ2[i] + (EJ2[i] + FJ2[i] / wave[midw]) / wave[midw]) / wave[midw]) / wave[midw]
        )

    hm_cx = np.zeros(nwave)
    for i in range(6):
        hm_cx += t_coeff ** ((i + 1) / 2.0) * hj[i, :]

    past20 = np.where(wave > 20.0)
    if np.size(past20) > 0:
        hm_cx[past20] = 0.0

    return hm_cx * 1.380658e-16 * t


def convert_cia_from_sqlite(
    sqlite_db: Path,
    output_dir: Path,
    marker_dir: Path,
    *,
    target_wno: np.ndarray | None = None,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    marker = marker_dir / "cia.done"
    extra = None
    if target_wno is not None:
        extra = {
            "target_wno_n": int(target_wno.size),
            "target_wno_min": float(np.min(target_wno)),
            "target_wno_max": float(np.max(target_wno)),
        }
    fingerprint = _source_fingerprint([sqlite_db], extra=extra)
    if not force and _marker_matches(marker, fingerprint):
        # If skipped, still need wno/T for follow-on stages.
        with sqlite3.connect(sqlite_db) as conn:
            cur = conn.cursor()
            db_wno = _decode_numpy_blob(cur.execute("SELECT wavenumber_grid FROM header LIMIT 1").fetchone()[0])
            temps = np.array(
                sorted({float(r[0]) for r in cur.execute("SELECT DISTINCT temperature FROM continuum")}),
                dtype=float,
            )
        return (target_wno if target_wno is not None else db_wno), temps

    map_names = {
        "H2H2": "H2-H2",
        "H2He": "H2-He",
        "H2H": "H2-H",
        "H2CH4": "H2-CH4",
        "H2N2": "H2-N2",
    }

    with sqlite3.connect(sqlite_db) as conn:
        cur = conn.cursor()
        db_wno = _decode_numpy_blob(cur.execute("SELECT wavenumber_grid FROM header LIMIT 1").fetchone()[0])
        rows = cur.execute("SELECT molecule, temperature, opacity FROM continuum").fetchall()

    cia_data: dict[str, list[tuple[float, np.ndarray]]] = {}
    for molecule, temp, blob in rows:
        if molecule not in map_names:
            continue
        xs_cm1_amagat2 = _decode_numpy_blob(blob)
        if xs_cm1_amagat2.size != db_wno.size:
            raise ValueError(f"Continuum row has wrong wavenumber size for molecule {molecule}")
        xs_cm5 = xs_cm1_amagat2 / (LOSCHMIDT_CM3 ** 2)
        if target_wno is not None:
            # Keep boundary values outside DB range to avoid artificial ultra-low tails.
            xs_cm5 = np.interp(target_wno, db_wno, xs_cm5)
        cia_data.setdefault(molecule, []).append((float(temp), xs_cm5))

    wno_out = target_wno if target_wno is not None else db_wno
    wavelengths = (1.0e4 / wno_out)[::-1]
    cout = output_dir / "CIA"
    cout.mkdir(parents=True, exist_ok=True)

    temps_sorted = np.array(sorted({x[0] for vals in cia_data.values() for x in vals}), dtype=float)

    for molecule, out_name in map_names.items():
        entries = sorted(cia_data.get(molecule, []), key=lambda x: x[0])
        if not entries:
            continue
        temp_vec = np.array([e[0] for e in entries], dtype=float)
        xs = np.vstack([e[1] for e in entries])  # (T,W)
        log10xs = np.log10(np.maximum(xs, SMALL)).T[::-1, :]  # (W,T), wavelength ascending
        notes = (
            f"{out_name} converted from PICASO continuum sqlite. "
            "wavelengths=[um], T=[K], log10xs=log10(cm^5/molecule^2)."
        )
        _write_h5_cia(cout / f"{out_name}.h5", wavelengths, temp_vec, log10xs, notes)

    _write_marker(marker, fingerprint)
    return wno_out, temps_sorted


def convert_special_hminus_cia(
    h2minus_csv: Path,
    wno: np.ndarray,
    temperatures: np.ndarray,
    output_dir: Path,
    marker_dir: Path,
    *,
    force: bool = False,
) -> int:
    marker = marker_dir / "cia_special.done"
    fingerprint = _source_fingerprint(
        [h2minus_csv],
        extra={
            "nwno": int(wno.size),
            "nt": int(temperatures.size),
            "wno_min": float(np.min(wno)),
            "wno_max": float(np.max(wno)),
        },
    )
    if not force and _marker_matches(marker, fingerprint):
        return 2

    theta, wno_bell, table = _load_h2minus_table(h2minus_csv)

    hminus_ff = np.zeros((temperatures.size, wno.size))
    h2minus = np.zeros((temperatures.size, wno.size))

    for i, temp in enumerate(temperatures):
        hminus_ff[i, :] = _get_hminusff(float(temp), wno)
        if temp < 600.0:
            h2minus_cm4_dyn = np.zeros(wno.size) + 1.0e-60
        else:
            h2minus_cm4_dyn = _get_h2minus(float(temp), wno, theta, wno_bell, table)
        # Convert cm^4/dyn -> cm^5/molecule^2 via k_B*T, matching PICASO usage.
        h2minus[i, :] = h2minus_cm4_dyn * BOLTZMANN_CGS * temp

    wavelengths = (1.0e4 / wno)[::-1]
    log10_hminus_ff = np.log10(np.maximum(hminus_ff, SMALL)).T[::-1, :]
    log10_h2minus = np.log10(np.maximum(h2minus, SMALL)).T[::-1, :]

    cout = output_dir / "CIA"
    _write_h5_cia(
        cout / "H--e-.h5",
        wavelengths,
        temperatures,
        log10_hminus_ff,
        "H--e- from PICASO get_hminusff (Bell & Berrington 1987). log10xs in cm^5/molecule^2.",
    )
    _write_h5_cia(
        cout / "H2--e-.h5",
        wavelengths,
        temperatures,
        log10_h2minus,
        "H2--e- from PICASO get_h2minus (Bell 1980), converted with k_B*T to cm^5/molecule^2.",
    )

    _write_marker(marker, fingerprint)
    return 2


def convert_hminus_bf_xsection(
    wno: np.ndarray,
    output_dir: Path,
    marker_dir: Path,
    *,
    force: bool = False,
) -> int:
    marker = marker_dir / "xsections.done"
    fingerprint = {"nwno": int(wno.size)}
    if not force and _marker_matches(marker, fingerprint):
        return 1

    wavelengths = (1.0e4 / wno)[::-1]
    photoabsorption = _get_hminusbf(wno)[::-1]
    zeros = np.zeros_like(photoabsorption)

    xout = output_dir / "xsections"
    _write_h5_photolysis_xsection(
        xout / "H-.h5",
        wavelengths,
        photoabsorption,
        zeros,
        zeros,
    )

    _write_marker(marker, fingerprint)
    return 1


def copy_rayleigh_opacities(
    rayleigh_source_dir: Path,
    output_dir: Path,
    marker_dir: Path,
    *,
    force: bool = False,
) -> int:
    src_files = sorted([p for p in rayleigh_source_dir.glob("*") if p.is_file()])
    if not src_files:
        raise FileNotFoundError(f"No rayleigh files found in {rayleigh_source_dir}")

    marker = marker_dir / "rayleigh.done"
    fingerprint = _source_fingerprint(src_files)
    if not force and _marker_matches(marker, fingerprint):
        return len(src_files)

    rout = output_dir / "rayleigh"
    rout.mkdir(parents=True, exist_ok=True)
    for src in src_files:
        shutil.copy2(src, rout / src.name)

    _write_marker(marker, fingerprint)
    return len(src_files)


def _validate_outputs(output_dir: Path) -> None:
    kdir = output_dir / "kdistributions"
    cdir = output_dir / "CIA"
    xdir = output_dir / "xsections"

    kfiles = sorted([p for p in kdir.glob("*.h5") if p.name != "bins.h5"])
    if not kfiles:
        raise RuntimeError("No converted k-distribution files found.")

    ref_wavelengths = None
    ref_weights = None
    for fpath in kfiles:
        with h5py.File(fpath, "r") as f:
            for key in ["T", "log10P", "wavelengths", "weights", "log10k"]:
                if key not in f:
                    raise RuntimeError(f"{fpath} missing dataset '{key}'")
            log10k = f["log10k"][:]
            if not np.isfinite(log10k).all():
                raise RuntimeError(f"{fpath} contains non-finite log10k")
            wav = f["wavelengths"][:]
            wts = f["weights"][:]
            if not np.all(np.diff(wav) > 0.0):
                raise RuntimeError(f"{fpath} has non-monotonic wavelengths")
            if ref_wavelengths is None:
                ref_wavelengths = wav
                ref_weights = wts
            else:
                if not np.allclose(ref_wavelengths, wav, rtol=0.0, atol=1e-8):
                    raise RuntimeError(f"Wavelength mismatch for {fpath}")
                if not np.allclose(ref_weights, wts, rtol=0.0, atol=1e-12):
                    raise RuntimeError(f"Weight mismatch for {fpath}")

    bins = kdir / "bins.h5"
    if not bins.exists():
        raise RuntimeError("kdistributions/bins.h5 is missing")
    with h5py.File(bins, "r") as f:
        for key in ["sol_wavl", "ir_wavl"]:
            if key not in f:
                raise RuntimeError(f"bins.h5 missing dataset '{key}'")
            if not np.all(np.diff(f[key][:]) > 0.0):
                raise RuntimeError(f"bins.h5/{key} is not monotonic ascending")

    cia_files = sorted(cdir.glob("*.h5"))
    if not cia_files:
        raise RuntimeError("No CIA files were produced.")
    for fpath in cia_files:
        with h5py.File(fpath, "r") as f:
            for key in ["wavelengths", "T", "log10xs"]:
                if key not in f:
                    raise RuntimeError(f"{fpath} missing dataset '{key}'")
            if not np.isfinite(f["log10xs"][:]).all():
                raise RuntimeError(f"{fpath} contains non-finite log10xs")

    xfile = xdir / "H-.h5"
    if not xfile.exists():
        raise RuntimeError("xsections/H-.h5 is missing")
    with h5py.File(xfile, "r") as f:
        for key in ["wavelengths", "photoabsorption", "photodissociation", "photoionisation"]:
            if key not in f:
                raise RuntimeError(f"{xfile} missing dataset '{key}'")
            if not np.isfinite(f[key][:]).all():
                raise RuntimeError(f"{xfile}:{key} has non-finite values")

    rayleigh_file = output_dir / "rayleigh" / "rayleigh.yaml"
    if not rayleigh_file.exists():
        raise RuntimeError(f"Missing rayleigh file: {rayleigh_file}")


def build_picaso_opacities(
    downloads_root: str | Path = "photochem_climate",
    picaso_downloads: str | Path = "photochem_climate/picaso_downloads",
    output_dir: str | Path = "photochem_climate/picaso_opacities",
    force: bool = False,
) -> Path:
    downloads_root_path = _resolve_path(downloads_root)
    picaso_downloads_path = _resolve_path(picaso_downloads)
    output_path = _resolve_path(output_dir)
    marker_dir = output_path / ".build_state"

    picaso_ref_dirs = sorted(picaso_downloads_path.glob("picaso-*/reference"))
    if not picaso_ref_dirs:
        raise FileNotFoundError(
            f"Could not find picaso reference folder under {picaso_downloads_path}. "
            "Expected picaso-*/reference/"
        )
    picaso_ref_dir = picaso_ref_dirs[0]

    sqlite_db = picaso_ref_dir / "climate_INPUTS" / "ck_cx_cont_opacities_661.db"
    h2minus_csv = picaso_ref_dir / "opacities" / "h2minus.csv"
    if not sqlite_db.exists():
        raise FileNotFoundError(f"Missing sqlite opacity DB: {sqlite_db}")
    if not h2minus_csv.exists():
        raise FileNotFoundError(f"Missing h2minus CSV: {h2minus_csv}")

    rayleigh_candidates = sorted(
        (downloads_root_path / "photochem_downloads").glob(
            "photochem_clima_data-*/photochem_clima_data/data/rayleigh"
        )
    )
    if not rayleigh_candidates:
        raise FileNotFoundError(
            f"Could not find photochem_clima_data rayleigh folder under {downloads_root_path / 'photochem_downloads'}"
        )
    rayleigh_source_dir = rayleigh_candidates[0]

    nk, k_wno = convert_kdistributions(picaso_downloads_path, output_path, marker_dir, force=force)
    wno, temps = convert_cia_from_sqlite(sqlite_db, output_path, marker_dir, target_wno=k_wno, force=force)
    nspecial = convert_special_hminus_cia(h2minus_csv, wno, temps, output_path, marker_dir, force=force)
    nxs = convert_hminus_bf_xsection(wno, output_path, marker_dir, force=force)
    nray = copy_rayleigh_opacities(rayleigh_source_dir, output_path, marker_dir, force=force)

    _validate_outputs(output_path)

    ncia = len(list((output_path / "CIA").glob("*.h5")))
    print("Opacity conversion complete")
    print(f"  kdistributions: {nk}")
    print(f"  CIA:            {ncia} (includes {nspecial} special H- sources)")
    print(f"  xsections:      {nxs}")
    print(f"  rayleigh files: {nray}")
    print(f"  output_dir:     {output_path}")

    return output_path


def main() -> None:
    local_downloads = CURRENT_DIR / "photochem_downloads"
    local_picaso_downloads = CURRENT_DIR / "picaso_downloads"
    local_output = CURRENT_DIR / "picaso_opacities"

    # First download/copy the stuff we need
    download_and_extract_zip(
        url="https://github.com/Nicholaswogan/photochem_clima_data/archive/refs/tags/v0.3.1.zip",
        extract_to=local_downloads,
    )
    download_and_extract_zip(
        url="https://github.com/Nicholaswogan/clima/archive/refs/tags/v0.7.3.zip",
        extract_to=local_downloads,
    )
    download_and_extract_zip(
        url="https://github.com/Nicholaswogan/photochem/archive/bdaade7895d495ffe53a5042ed2f4d91a114d6e0.zip",
        extract_to=local_downloads,
    )
    download_and_extract_zip(
        url="https://github.com/natashabatalha/picaso/archive/3c322c9d1b254a0f65067b0a7a9b1fc8aec73981.zip",
        extract_to=local_picaso_downloads,
    )
    download_and_extract_zip(
        url="https://zenodo.org/api/records/18644980/files-archive",
        extract_to=local_picaso_downloads,
    )

    build_picaso_opacities(
        downloads_root=CURRENT_DIR,
        picaso_downloads=local_picaso_downloads,
        output_dir=local_output,
        force=False,
    )


if __name__ == "__main__":
    main()
