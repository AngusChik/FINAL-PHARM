# Locally hosted browser dependencies

These pinned files are committed so the LAN application keeps working without
internet access. Runtime templates must reference these local copies only.

| Package | Version | Local file | Upstream source | License |
| --- | --- | --- | --- | --- |
| Chart.js | 4.4.0 | `chartjs/chart.umd.min.js` | `https://www.npmjs.com/package/chart.js/v/4.4.0` | MIT (`chartjs/LICENSE.md`) |
| flatpickr | 4.6.13 | `flatpickr/flatpickr.min.js`, `flatpickr/flatpickr.min.css` | `https://www.npmjs.com/package/flatpickr/v/4.6.13` | MIT (`flatpickr/LICENSE.md`) |
| jsPDF | 2.5.1 | `jspdf/jspdf.umd.min.js` | `https://www.npmjs.com/package/jspdf/v/2.5.1` | MIT (`jspdf/LICENSE-jspdf.txt`) |
| jsPDF-AutoTable | 3.8.4 | `jspdf/jspdf.plugin.autotable.min.js` | `https://www.npmjs.com/package/jspdf-autotable/v/3.8.4` | MIT (`jspdf/LICENSE-autotable.txt`) |
| Libre Barcode 128 | Google Fonts snapshot downloaded 2026-08-14 | `libre-barcode-128/libre-barcode-128.woff2` | `https://fonts.google.com/specimen/Libre+Barcode+128` | SIL OFL 1.1 (`libre-barcode-128/OFL.txt`) |

## SHA-256 checksums

```text
0e2326c6868072bec1592760c6729043caeea2960a2b46cee6a2192aac6abff0  chartjs/chart.umd.min.js
1b34a42552c96f10e4dfaaa4a367276b03868aacff63c1ac42ffe331352bc754  flatpickr/flatpickr.min.css
1eeab1cb779471a0b0aaa93dd91c2eb1aa537d696f01ab05ea9dabc55e8525a1  flatpickr/flatpickr.min.js
2223830cf9a1ec85af014cc71b37c1b1eb566f3d18b2ab8071e96af822c58bdb  jspdf/jspdf.plugin.autotable.min.js
98ccf17aa10c20bb1301762618fcc9b6ab3a4e7f26b6071d64d0b41154df3875  jspdf/jspdf.umd.min.js
08354cde6099d600b6963e40eeade0e78ff6ae950c4f87578c1190fb1dc20ec6  libre-barcode-128/libre-barcode-128.woff2
```
