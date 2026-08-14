(function (global) {
    'use strict';

    function asText(value) {
        return value === null || value === undefined ? '' : String(value).trim();
    }

    function normalizeBarcode(value) {
        var raw = asText(value).toLowerCase();
        var compact = raw.replace(/[\s-]/g, '');
        if (/^\d+$/.test(compact)) {
            return compact.replace(/^0+/, '') || '0';
        }
        return compact;
    }

    function matchingProducts(products, query, limit) {
        var raw = asText(query).toLowerCase();
        if (!raw) return [];
        var normalized = normalizeBarcode(raw);
        var maximum = Number(limit) > 0 ? Number(limit) : 10;

        return (products || []).filter(function (product) {
            var name = asText(product.name).toLowerCase();
            var itemNumber = asText(product.item_number).toLowerCase();
            var barcode = asText(product.barcode).toLowerCase();
            var normalizedBarcode = normalizeBarcode(product.barcode);
            return name.indexOf(raw) !== -1
                || itemNumber.indexOf(raw) !== -1
                || barcode.indexOf(raw) !== -1
                || Boolean(normalized && normalizedBarcode
                    && normalizedBarcode.indexOf(normalized) !== -1);
        }).slice(0, maximum);
    }

    function exactBarcodeProduct(products, query) {
        var normalized = normalizeBarcode(query);
        if (!normalized) return null;
        return (products || []).find(function (product) {
            return Boolean(product.barcode)
                && normalizeBarcode(product.barcode) === normalized;
        }) || null;
    }

    function isLikelyBarcode(value) {
        return /^\d{4,}$/.test(asText(value).replace(/[\s-]/g, ''));
    }

    global.ProductLookup = Object.freeze({
        exactBarcodeProduct: exactBarcodeProduct,
        isLikelyBarcode: isLikelyBarcode,
        matchingProducts: matchingProducts,
        normalizeBarcode: normalizeBarcode,
    });
}(window));
