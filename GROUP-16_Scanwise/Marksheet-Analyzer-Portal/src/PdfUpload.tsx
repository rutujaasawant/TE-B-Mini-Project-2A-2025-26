import { useState } from "react";
import {API_URL} from "../config"

const BASE_URL = API_URL;

const HomePage = ({ setPage }: { setPage: (page: string) => void }) => {
  return (
    <div
      style={{
        background: "#ffffff",
        minHeight: "100vh",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
        color: "#333",
        padding: "20px",
        margin: "0",
      }}
    >
      <h1
        style={{
          textAlign: "center",
          marginBottom: "50px",
          fontSize: "3rem",
          color: "#E5322D",
          fontWeight: "700",
        }}
      >
        ScanWise
      </h1>
      <div style={{ display: "flex", flexDirection: "row", gap: "30px", justifyContent: "center" }}>
        <button
          onClick={() => setPage("pdf-analysis")}
          style={{
            background: "#E5322D",
            color: "#ffffff",
            border: "none",
            borderRadius: "6px",
            padding: "24px 50px",
            fontWeight: "600",
            fontSize: "1.2rem",
            cursor: "pointer",
            boxShadow: "0 2px 8px rgba(229,50,45,0.2)",
            transition: "all 0.3s ease",
            minWidth: "250px",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "translateY(-2px)";
            e.currentTarget.style.boxShadow = "0 4px 12px rgba(229,50,45,0.3)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "0 2px 8px rgba(229,50,45,0.2)";
          }}
        >
          PDF Analysis
        </button>
        <button
          onClick={() => setPage("excel-analysis")}
          style={{
            background: "#E5322D",
            color: "#ffffff",
            border: "none",
            borderRadius: "6px",
            padding: "24px 50px",
            fontWeight: "600",
            fontSize: "1.2rem",
            cursor: "pointer",
            boxShadow: "0 2px 8px rgba(229,50,45,0.2)",
            transition: "all 0.3s ease",
            minWidth: "250px",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "translateY(-2px)";
            e.currentTarget.style.boxShadow = "0 4px 12px rgba(229,50,45,0.3)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "0 2px 8px rgba(229,50,45,0.2)";
          }}
        >
          Excel Analysis
        </button>
      </div>
    </div>
  );
};

const PdfAnalysisPage = ({ setPage }: { setPage: (page: string) => void }) => {
  const [mode, setMode] = useState<null | "single" | "multiple">(null);
  const [singlePdf, setSinglePdf] = useState<File | null>(null);
  const [sem1Pdf, setSem1Pdf] = useState<File | null>(null);
  const [sem2Pdf, setSem2Pdf] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [showJson, setShowJson] = useState(false);
  
  interface PdfResults {
    data: any[];
    jsonFile: string;
    excelFile: string;
  }
  const [results, setResults] = useState<PdfResults | null>(null);

  const handleSinglePdfSubmit = async () => {
    if (!singlePdf) {
      alert("Please select a PDF file.");
      return;
    }

    const formData = new FormData();
    formData.append("marksheet", singlePdf);

    try {
      setLoading(true);
      const res = await fetch(
        `${BASE_URL}/analysis/get-single-pdf-percentage-analysis-data/`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) throw new Error("Failed to upload file");

      const data = await res.json();
      if (data.success) {
        setResults({
          data: data.results,
          jsonFile: `${BASE_URL}${data.json_file}`,
          excelFile: `${BASE_URL}${data.excel_file}`,
        });
      } else {
        alert(data.message || "Something went wrong.");
      }
    } catch (err) {
      if (err instanceof Error) {
        alert(err.message);
      } else {
        alert("An unknown error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleMultiplePdfSubmit = async () => {
    if (!sem1Pdf || !sem2Pdf) {
      alert("Please select both PDF files.");
      return;
    }

    const formData = new FormData();
    formData.append("sem1_pdf", sem1Pdf);
    formData.append("sem2_pdf", sem2Pdf);

    try {
      setLoading(true);
      const res = await fetch(
        `${BASE_URL}/analysis/get-multiple-pdf-percentage-analysis-data/`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) throw new Error("Failed to upload files");

      const data = await res.json();
      if (data.success) {
        setResults({
          data: data.results,
          jsonFile: `${BASE_URL}${data.json_file}`,
          excelFile: `${BASE_URL}${data.excel_file}`,
        });
      } else {
        alert(data.message || "Something went wrong.");
      }
    } catch (err) {
      if (err instanceof Error) {
        alert(err.message);
      } else {
        alert("An unknown error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        background: "#ffffff",
        minHeight: "100vh",
        width: "100%",
        padding: "40px 60px 60px",
        fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
        color: "#333",
        margin: "0",
      }}
    >
      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
        }}
      >
        <button
          onClick={() => setPage("home")}
          style={{
            background: "transparent",
            color: "#E5322D",
            border: "2px solid #E5322D",
            borderRadius: "6px",
            padding: "10px 20px",
            marginBottom: "30px",
            cursor: "pointer",
            fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
            fontWeight: "600",
            transition: "all 0.3s ease",
            fontSize: "1rem",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = "#E5322D";
            e.currentTarget.style.color = "#ffffff";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "#E5322D";
          }}
        >
          ← Back
        </button>

        <h2
          style={{
            textAlign: "center",
            marginBottom: "40px",
            color: "#E5322D",
            fontWeight: "700",
            fontSize: "2rem",
          }}
        >
          PDF Analysis
        </h2>

        {!mode && (
          <div
            style={{ display: "flex", flexDirection: "row", gap: "24px", justifyContent: "center" }}
          >
            <button
              onClick={() => setMode("single")}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "20px 40px",
                fontWeight: "600",
                fontSize: "1.1rem",
                cursor: "pointer",
                transition: "all 0.3s ease",
                minWidth: "250px",
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = "#c72a26";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = "#E5322D";
              }}
            >
              Single PDF Analysis
            </button>
            <button
              onClick={() => setMode("multiple")}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "20px 40px",
                fontWeight: "600",
                fontSize: "1.1rem",
                cursor: "pointer",
                transition: "all 0.3s ease",
                minWidth: "250px",
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = "#c72a26";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = "#E5322D";
              }}
            >
              Multiple PDF Analysis
            </button>
          </div>
        )}

        {mode === "single" && (
          <div style={{ maxWidth: "600px", margin: "0 auto" }}>
            <label
              style={{
                display: "block",
                marginBottom: "10px",
                fontWeight: "600",
                color: "#555",
                fontSize: "1rem",
              }}
            >
              Select PDF File:
            </label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) =>
                setSinglePdf(
                  e.target.files && e.target.files[0] ? e.target.files[0] : null
                )
              }
              style={{
                background: "#f7f7f7",
                color: "#333",
                border: "2px solid #ddd",
                borderRadius: "6px",
                padding: "12px",
                width: "100%",
                marginBottom: "20px",
                fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
                fontSize: "1rem",
              }}
            />
            <button
              onClick={handleSinglePdfSubmit}
              disabled={loading}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "14px",
                fontWeight: "600",
                width: "100%",
                cursor: loading ? "not-allowed" : "pointer",
                opacity: loading ? 0.6 : 1,
                fontSize: "1rem",
              }}
            >
              {loading ? "Analyzing..." : "Submit"}
            </button>
            <button
              onClick={() => {
                setMode(null);
                setResults(null);
                setShowJson(false);
              }}
              style={{
                background: "transparent",
                color: "#E5322D",
                border: "2px solid #E5322D",
                borderRadius: "6px",
                padding: "14px",
                marginTop: "12px",
                width: "100%",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "1rem",
              }}
            >
              Cancel
            </button>
          </div>
        )}

        {mode === "multiple" && (
          <div style={{ maxWidth: "600px", margin: "0 auto" }}>
            <label
              style={{
                display: "block",
                marginBottom: "10px",
                fontWeight: "600",
                color: "#555",
                fontSize: "1rem",
              }}
            >
              Semester 1 PDF:
            </label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) =>
                setSem1Pdf(
                  e.target.files && e.target.files[0] ? e.target.files[0] : null
                )
              }
              style={{
                background: "#f7f7f7",
                color: "#333",
                border: "2px solid #ddd",
                borderRadius: "6px",
                padding: "12px",
                width: "100%",
                marginBottom: "20px",
                fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
                fontSize: "1rem",
              }}
            />
            <label
              style={{
                display: "block",
                marginBottom: "10px",
                fontWeight: "600",
                color: "#555",
                fontSize: "1rem",
              }}
            >
              Semester 2 PDF:
            </label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) =>
                setSem2Pdf(
                  e.target.files && e.target.files[0] ? e.target.files[0] : null
                )
              }
              style={{
                background: "#f7f7f7",
                color: "#333",
                border: "2px solid #ddd",
                borderRadius: "6px",
                padding: "12px",
                width: "100%",
                marginBottom: "20px",
                fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
                fontSize: "1rem",
              }}
            />
            <button
              onClick={handleMultiplePdfSubmit}
              disabled={loading}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "14px",
                fontWeight: "600",
                width: "100%",
                cursor: loading ? "not-allowed" : "pointer",
                opacity: loading ? 0.6 : 1,
                fontSize: "1rem",
              }}
            >
              {loading ? "Analyzing..." : "Submit"}
            </button>
            <button
              onClick={() => {
                setMode(null);
                setResults(null);
                setShowJson(false);
              }}
              style={{
                background: "transparent",
                color: "#E5322D",
                border: "2px solid #E5322D",
                borderRadius: "6px",
                padding: "14px",
                marginTop: "12px",
                width: "100%",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "1rem",
              }}
            >
              Cancel
            </button>
          </div>
        )}

        {results && (
          <div
            style={{
              marginTop: "30px",
              padding: "20px",
              background: "#f7f7f7",
              borderRadius: "6px",
              border: "1px solid #ddd",
              maxWidth: "600px",
              margin: "30px auto 0",
            }}
          >
            <div style={{ display: "flex", gap: "16px" }}>
              <button
                onClick={() => setShowJson(!showJson)}
                style={{
                  flex: 1,
                  textAlign: "center",
                  background: "#E5322D",
                  color: "#ffffff",
                  padding: "14px",
                  borderRadius: "6px",
                  border: "none",
                  fontWeight: "600",
                  cursor: "pointer",
                  transition: "all 0.3s ease",
                  fontSize: "1rem",
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = "#c72a26";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = "#E5322D";
                }}
              >
                {showJson ? "Hide JSON" : "View JSON"}
              </button>
              <a
                href={results.excelFile}
                download
                style={{
                  flex: 1,
                  textAlign: "center",
                  background: "#E5322D",
                  color: "#fff",
                  padding: "14px",
                  borderRadius: "6px",
                  textDecoration: "none",
                  fontWeight: "600",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "all 0.3s ease",
                  fontSize: "1rem",
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = "#c72a26";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = "#E5322D";
                }}
              >
                Download Excel
              </a>
            </div>

            {showJson && (
              <div
                style={{
                  marginTop: "16px",
                  background: "#ffffff",
                  borderRadius: "6px",
                  padding: "16px",
                  maxHeight: "400px",
                  overflowY: "auto",
                  border: "1px solid #ddd",
                }}
              >
                <pre
                  style={{
                    color: "#333",
                    fontSize: "12px",
                    margin: 0,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                  }}
                >
                  {JSON.stringify(results.data, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const ExcelAnalysisPage = ({
  setPage,
}: {
  setPage: (page: string) => void;
}) => {
  const [mode, setMode] = useState<null | "kt" | "passfail" | "average">(null);
  const [excelFile, setExcelFile] = useState<File | null>(null);
  const [excelFiles, setExcelFiles] = useState<{
    [key: string]: File | undefined;
  }>({});
  const [loading, setLoading] = useState(false);
  interface ExcelResults {
    excelFile: string;
    chartUrl?: string;
    chartData?: any;
  }
  const [results, setResults] = useState<ExcelResults | null>(null);

  const handleKTStudents = async () => {
    if (!excelFile) {
      alert("Please select an Excel file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", excelFile);

    try {
      setLoading(true);
      const res = await fetch(`${BASE_URL}/analysis/get-kt-students/`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to upload file");

      const data = await res.json();
      setResults({ excelFile: `${BASE_URL}${data.excel_file}` });
    } catch (err) {
      if (err instanceof Error) {
        alert(err.message);
      } else {
        alert("An unknown error occurred.");
      }
    }
    finally {
        setLoading(false);
    }
  };

  const handlePassFailAnalysis = async () => {
    if (!excelFile) {
      alert("Please select an Excel file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", excelFile);

    try {
      setLoading(true);
      const res = await fetch(`${BASE_URL}/analysis/pass-fail-analysis/`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to upload file");

      const data = await res.json();
      
      setResults({
        excelFile: "",
        chartUrl: `${BASE_URL}${data.chart_url}`,
        chartData: data.chart_data,
      });
    } catch (err) {
      if (err instanceof Error) {
        alert(err.message);
      } else {
        alert("An unknown error occurred.");
      }
    }
    finally {
        setLoading(false);
    }
  };

  const handleAveragePercentages = async () => {
    const fileCount = Object.keys(excelFiles).filter(
      (key) => excelFiles[key]
    ).length;
    if (fileCount === 0) {
      alert("Please select at least one Excel file.");
      return;
    }

    const formData = new FormData();
    for (let i = 1; i <= 8; i++) {
      if (excelFiles[`file${i}`]) {
        if (excelFiles[`file${i}`]) {
          formData.append(`file${i}`, excelFiles[`file${i}`] as File);
        }
      }
    }

    try {
      setLoading(true);
      const res = await fetch(`${BASE_URL}/analysis/average-semesters/`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to upload files");

      const data = await res.json();
      
      setResults({ excelFile: `${BASE_URL}${data.excel_file}` });
    } catch (err) {
      if (err instanceof Error) {
        alert(err.message);
      } else {
        alert("An unknown error occurred.");
      }
    }
    finally {
        setLoading(false);
    }
  };

  return (
    <div
      style={{
        background: "#ffffff",
        minHeight: "100vh",
        width: "100%",
        padding: "40px 60px 60px",
        fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
        color: "#333",
        margin: "0",
      }}
    >
      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
        }}
      >
        <button
          onClick={() => setPage("home")}
          style={{
            background: "transparent",
            color: "#E5322D",
            border: "2px solid #E5322D",
            borderRadius: "6px",
            padding: "10px 20px",
            marginBottom: "30px",
            cursor: "pointer",
            fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
            fontWeight: "600",
            transition: "all 0.3s ease",
            fontSize: "1rem",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = "#E5322D";
            e.currentTarget.style.color = "#ffffff";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "#E5322D";
          }}
        >
          ← Back
        </button>

        <h2
          style={{
            textAlign: "center",
            marginBottom: "40px",
            color: "#E5322D",
            fontWeight: "700",
            fontSize: "2rem",
          }}
        >
          Excel Analysis
        </h2>

        {!mode && (
          <div
            style={{ display: "flex", flexDirection: "row", gap: "24px", justifyContent: "center", flexWrap: "wrap" }}
          >
            <button
              onClick={() => setMode("kt")}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "20px 40px",
                fontWeight: "600",
                fontSize: "1.1rem",
                cursor: "pointer",
                transition: "all 0.3s ease",
                minWidth: "220px",
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = "#c72a26";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = "#E5322D";
              }}
            >
              Get KT Students
            </button>
            <button
              onClick={() => setMode("passfail")}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "20px 40px",
                fontWeight: "600",
                fontSize: "1.1rem",
                cursor: "pointer",
                transition: "all 0.3s ease",
                minWidth: "220px",
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = "#c72a26";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = "#E5322D";
              }}
            >
              Pass/Fail Analysis
            </button>
            <button
              onClick={() => setMode("average")}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "20px 40px",
                fontWeight: "600",
                fontSize: "1.1rem",
                cursor: "pointer",
                transition: "all 0.3s ease",
                minWidth: "220px",
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = "#c72a26";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = "#E5322D";
              }}
            >
              Average Percentages
            </button>
          </div>
        )}

        {mode === "kt" && (
          <div style={{ maxWidth: "600px", margin: "0 auto" }}>
            <label
              style={{
                display: "block",
                marginBottom: "10px",
                fontWeight: "600",
                color: "#555",
                fontSize: "1rem",
              }}
            >
              Select Excel File:
            </label>
            <input
              type="file"
              accept=".xlsx,.xls"
              onChange={(e) =>
                setExcelFile(
                  e.target.files && e.target.files[0] ? e.target.files[0] : null
                )
              }
              style={{
                background: "#f7f7f7",
                color: "#333",
                border: "2px solid #ddd",
                borderRadius: "6px",
                padding: "12px",
                width: "100%",
                marginBottom: "20px",
                fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
                fontSize: "1rem",
              }}
            />
            <button
              onClick={handleKTStudents}
              disabled={loading}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "14px",
                fontWeight: "600",
                width: "100%",
                cursor: loading ? "not-allowed" : "pointer",
                opacity: loading ? 0.6 : 1,
                fontSize: "1rem",
              }}
            >
              {loading ? "Processing..." : "Submit"}
            </button>
            <button
              onClick={() => {
                setMode(null);
                setResults(null);
              }}
              style={{
                background: "transparent",
                color: "#E5322D",
                border: "2px solid #E5322D",
                borderRadius: "6px",
                padding: "14px",
                marginTop: "12px",
                width: "100%",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "1rem",
              }}
            >
              Cancel
            </button>
          </div>
        )}

        {mode === "passfail" && (
          <div style={{ maxWidth: "600px", margin: "0 auto" }}>
            <label
              style={{
                display: "block",
                marginBottom: "10px",
                fontWeight: "600",
                color: "#555",
                fontSize: "1rem",
              }}
            >
              Select Excel File:
            </label>
            <input
              type="file"
              accept=".xlsx,.xls"
              onChange={(e) =>
                setExcelFile(
                  e.target.files && e.target.files[0] ? e.target.files[0] : null
                )
              }
              style={{
                background: "#f7f7f7",
                color: "#333",
                border: "2px solid #ddd",
                borderRadius: "6px",
                padding: "12px",
                width: "100%",
                marginBottom: "20px",
                fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
                fontSize: "1rem",
              }}
            />
            <button
              onClick={handlePassFailAnalysis}
              disabled={loading}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "14px",
                fontWeight: "600",
                width: "100%",
                cursor: loading ? "not-allowed" : "pointer",
                opacity: loading ? 0.6 : 1,
                fontSize: "1rem",
              }}
            >
              {loading ? "Processing..." : "Submit"}
            </button>
            <button
              onClick={() => {
                setMode(null);
                setResults(null);
              }}
              style={{
                background: "transparent",
                color: "#E5322D",
                border: "2px solid #E5322D",
                borderRadius: "6px",
                padding: "14px",
                marginTop: "12px",
                width: "100%",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "1rem",
              }}
            >
              Cancel
            </button>
          </div>
        )}

        {mode === "average" && (
          <div style={{ maxWidth: "700px", margin: "0 auto" }}>
            <p style={{ marginBottom: "20px", fontSize: "1rem", color: "#666", textAlign: "center" }}>
              Upload 1-8 semester files (at least one required):
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              {[1, 2, 3, 4, 5, 6, 7, 8].map((num) => (
                <div key={num}>
                  <label
                    style={{
                      display: "block",
                      marginBottom: "6px",
                      fontSize: "0.95rem",
                      fontWeight: "600",
                      color: "#555",
                    }}
                  >
                    Semester {num}:
                  </label>
                  <input
                    type="file"
                    accept=".xlsx,.xls"
                    onChange={(e) =>
                      setExcelFiles({
                        ...excelFiles,
                        [`file${num}`]:
                          e.target.files && e.target.files[0]
                            ? e.target.files[0]
                            : undefined,
                      })
                    }
                    style={{
                      background: "#f7f7f7",
                      color: "#333",
                      border: "2px solid #ddd",
                      borderRadius: "6px",
                      padding: "10px",
                      width: "100%",
                      fontSize: "0.9rem",
                      fontFamily: "'Open Sans', 'Segoe UI', Arial, sans-serif",
                    }}
                  />
                </div>
              ))}
            </div>
            <button
              onClick={handleAveragePercentages}
              disabled={loading}
              style={{
                background: "#E5322D",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "14px",
                fontWeight: "600",
                width: "100%",
                marginTop: "24px",
                cursor: loading ? "not-allowed" : "pointer",
                opacity: loading ? 0.6 : 1,
                fontSize: "1rem",
              }}
            >
              {loading ? "Processing..." : "Submit"}
            </button>
            <button
              onClick={() => {
                setMode(null);
                setResults(null);
              }}
              style={{
                background: "transparent",
                color: "#E5322D",
                border: "2px solid #E5322D",
                borderRadius: "6px",
                padding: "14px",
                marginTop: "12px",
                width: "100%",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "1rem",
              }}
            >
              Cancel
            </button>
          </div>
        )}

        {results && results.excelFile && (
          <div style={{ marginTop: "30px", textAlign: "center", paddingBottom: "40px" }}>
            <a
              href={results.excelFile}
              download
              style={{
                display: "inline-block",
                background: "#E5322D",
                color: "#ffffff",
                padding: "14px 32px",
                borderRadius: "6px",
                textDecoration: "none",
                fontWeight: "600",
                transition: "all 0.3s ease",
                fontSize: "1rem",
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = "#c72a26";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = "#E5322D";
              }}
            >
              Download Excel Results
            </a>
          </div>
        )}

        {results && results.chartUrl && (
          <div style={{ marginTop: "30px", maxWidth: "900px", margin: "30px auto 40px" }}>
            <h3 style={{ marginBottom: "20px", color: "#333", fontWeight: "600", fontSize: "1.3rem" }}>
              Pass/Fail Chart:
            </h3>
            <img
              src={results.chartUrl}
              alt="Pass/Fail Chart"
              style={{
                width: "100%",
                borderRadius: "6px",
                marginBottom: "20px",
                border: "1px solid #ddd",
              }}
            />
            <div
              style={{
                background: "#f7f7f7",
                padding: "20px",
                borderRadius: "6px",
                fontSize: "0.95rem",
                border: "1px solid #ddd",
              }}
            >
              {results.chartData.courses.map((course: string, idx: number) => (
                <div key={idx} style={{ marginBottom: "10px", color: "#555" }}>
                  <strong style={{ color: "#333" }}>{course}:</strong> Pass:{" "}
                  {results.chartData.pass_counts[idx]}, Fail:{" "}
                  {results.chartData.fail_counts[idx]}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const App = () => {
  const [page, setPage] = useState("home");

  return (
    <>
      {page === "home" && <HomePage setPage={setPage} />}
      {page === "pdf-analysis" && <PdfAnalysisPage setPage={setPage} />}
      {page === "excel-analysis" && <ExcelAnalysisPage setPage={setPage} />}
    </>
  );
};

export default App;
