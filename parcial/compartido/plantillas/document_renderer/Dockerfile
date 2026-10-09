FROM python:3.12-slim@sha256:7026274c107626d7e940e0e5d6730481a4600ae95d5ca7eb532dd4180313fea9

# The immutable Debian snapshot freezes LibreOffice and font package versions.
RUN printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260301T000000Z trixie main\n' > /etc/apt/sources.list \
    && rm -f /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends libreoffice-writer libreoffice-math \
       fonts-lmodern fonts-liberation2 fonts-dejavu-core poppler-utils \
    && dpkg-query -W > /opt/document-renderer-packages.tsv \
    && fc-list | sort > /opt/document-renderer-fonts.txt \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC
RUN pip install --no-cache-dir PyMuPDF==1.26.7 latex2mathml==3.78.1 pypdf==6.7.1 Pillow==12.1.1
RUN apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends python3-uno \
    && dpkg-query -W > /opt/document-renderer-packages.tsv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
