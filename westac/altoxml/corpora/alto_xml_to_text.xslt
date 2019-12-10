<?xml version="1.0"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >
<xsl:output method="text" encoding="utf-8"/>

<xsl:strip-space elements="*" />

<xsl:param name="postags"/>
<xsl:param name="deliminator"/>
<xsl:param name="target"/>
<xsl:param name="ignores" select="'|MAD|MID|PAD|'"/>

<xsl:template match="w">

  <xsl:if test="@pos!='MAD' and @pos!='MID' and @pos!='PAD'">
    <xsl:text> </xsl:text>
  </xsl:if>
  
  <xsl:value-of select="text()"/>

</xsl:template>
</xsl:stylesheet>